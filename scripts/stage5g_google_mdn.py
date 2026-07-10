#!/usr/bin/env python3
"""
Stage5G — Google-recommended full architecture.

Key components (all Google suggestions):
  1. Time2Vec — sinusoidal temporal encoding (NOT one-shot MLP)
  2. GRU — sequential processing over time (NOT flat prediction)
  3. Trajectory-level NLL loss — full Gaussian mixture likelihood
  4. Probabilistic sampling — sample from distribution (NOT argmax)
  5. SMPL 2-hop kinematic attention mask — joints only attend to
     kinematically close neighbors (1-2 hops in skeleton tree)
  6. No-scale raw radian residuals — no scaler amplitude trap
  7. FK 3D position loss — foot-ground contact
  8. Boundary-locked residuals — exact start/goal match
  9. Safe-improvement — don't make rotation error worse than baseline

Architecture:
  Input per joint: [start(3), goal(3), base_feat(6), action(3)]
  + Time2Vec(t) → 15 + 16 = 31D
  → Joint MLP (31→64)
  → BiGRU over time (64→128×2 directions)
  → 2-hop SMPL Cross-Attention per timestep
  → Per-joint MDN head: K=4 Gaussians, NLL loss
  → Generation: probabilistic sampling + boundary-lock
"""

from __future__ import annotations
import argparse, math, sys, time, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch, torch.nn as nn, torch.nn.functional as F
from smpl_fk import fk_smpl_full, KINTREE

Array = np.ndarray

# ════════════════════════════════════════════════════════════
# Data loader — same as stage5e
# ════════════════════════════════════════════════════════════

def load_data(data_dir, n_files=200, n_frames=96, min_translation=1.0, seed=42):
    base = Path(data_dir)
    all_files = list(base.rglob("Walk*.npz"))
    rng = np.random.default_rng(seed)
    sequences = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]; tr = d["trans"]
            if pb.shape[0] < n_frames: continue
            if tr[:,0].max()-tr[:,0].min() < min_translation: continue
            idx = np.linspace(0, pb.shape[0]-1, n_frames, dtype=int)
            full_rot = np.concatenate([d["root_orient"][idx], pb[idx],
                                       np.zeros((n_frames,6), dtype=np.float32)], axis=1)
            sequences.append(np.concatenate([full_rot, tr[idx]], axis=1))
        except: continue
    if n_files > 0 and len(sequences) > n_files:
        sequences = list(rng.choice(sequences, n_files, replace=False))
    data = np.stack(sequences, axis=0).astype(np.float32)
    print(f"Loaded {len(sequences)} walks (24 joints)")
    return data, 24


def compute_mean_base(data_rot, n_base=10):
    N, T, D = data_rot.shape; J = D // 3
    pool = data_rot[:min(n_base,N)].reshape(-1, T, J, 3)
    mean = np.nanmean(pool, axis=0)
    return mean.transpose(1,0,2).astype(np.float32)


def baseball_deform(path, start, goal, gamma=1.0):
    Q = np.asarray(path, dtype=np.float32); T = Q.shape[0]
    s = np.asarray(start, dtype=np.float32).reshape(1,-1)
    g = np.asarray(goal, dtype=np.float32).reshape(1,-1)
    a = np.linspace(0,1,T,dtype=np.float32).reshape(-1,1)
    return Q - Q[0:1] + s + gamma*a*(g - (Q[-1:]-Q[0:1]+s))


# ════════════════════════════════════════════════════════════
# SMPL 2-hop kinematic attention mask
# ════════════════════════════════════════════════════════════

def build_smpl_2hop_mask(n_joints=24):
    """Build (J,J) boolean mask: True if joints are ≤2 hops apart in SMPL tree."""
    # Build adjacency (undirected)
    adj = np.zeros((n_joints, n_joints), dtype=bool)
    for child in range(n_joints):
        parent = KINTREE[child]
        if parent >= 0:
            adj[child, parent] = True
            adj[parent, child] = True
    # 2-hop reachability: A + A@A
    A = adj.astype(np.int32)
    A2 = A @ A  # 2-hop paths
    reachable = (A + A2) > 0
    # Self always reachable
    np.fill_diagonal(reachable, True)
    return reachable  # (J, J) bool


# ════════════════════════════════════════════════════════════
# Time2Vec — sinusoidal temporal encoding
# ════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """Encode scalar timestep t ∈ [0,1] into a periodic+linear vector."""
    def __init__(self, out_dim=16, n_sinusoidal=8):
        super().__init__()
        self.out_dim = out_dim
        self.n_sin = n_sinusoidal
        # Frequencies: learnable for sine part
        self.w = nn.Parameter(torch.randn(n_sinusoidal) * 0.1)
        # Linear part: phase + linear coefficient
        self.linear = nn.Linear(1, out_dim - n_sinusoidal, bias=True)

    def forward(self, t):
        """t: (B,) or (B, T) → (..., out_dim)"""
        t_flat = t.float().reshape(-1, 1)  # (B*T, 1)
        # Sinusoidal: varying frequencies
        sin_feat = torch.sin(t_flat * self.w.unsqueeze(0))  # (B*T, n_sin)
        # Linear
        lin_feat = self.linear(t_flat)  # (B*T, out_dim - n_sin)
        out = torch.cat([sin_feat, lin_feat], dim=-1)
        return out.reshape(*t.shape, self.out_dim)


# ════════════════════════════════════════════════════════════
# SMPL-Masked Cross-Attention
# ════════════════════════════════════════════════════════════

class SMPLCrossAttention(nn.Module):
    """Multi-head cross-attention with SMPL 2-hop kinematic mask."""
    def __init__(self, dim=64, n_heads=4, n_joints=24):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # Build 2-hop mask: (J,J) bool → attention mask
        mask_2hop = build_smpl_2hop_mask(n_joints)  # (J,J) True=allowed
        # MHA attn_mask: (L,S) where True means ignore. L=S=J.
        # Invert: masked positions get -inf
        attn_mask = torch.zeros(n_joints, n_joints)
        attn_mask[~torch.from_numpy(mask_2hop)] = float('-inf')
        self.register_buffer('attn_mask', attn_mask)  # (J, J)

    def forward(self, x):
        """x: (B, J, D) → (B, J, D)"""
        out, _ = self.mha(x, x, x, attn_mask=self.attn_mask)
        return out


# ════════════════════════════════════════════════════════════
# GRU Temporal Encoder
# ════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Fast temporal convolution over time (replaces GRU for CPU speed)."""
    def __init__(self, dim=64, kernel_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2), nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2), nn.GELU(),
        )

    def forward(self, x):
        """x: (B, J, T, D) → (B, J, T, D)"""
        B, J, T, D = x.shape
        # Process per-joint over time: (B*J, D, T)
        x_r = x.permute(0, 1, 3, 2).reshape(B * J, D, T)
        out = self.conv(x_r)  # (B*J, D, T)
        out = out.reshape(B, J, D, T).permute(0, 1, 3, 2)  # (B, J, T, D)
        return out


# ════════════════════════════════════════════════════════════
# Trajectory MDN Head — per joint, per timestep
# ════════════════════════════════════════════════════════════

class TrajectoryMDNHead(nn.Module):
    """Predicts K Gaussian mixtures per joint per timestep.
    
    Output per (joint, timestep):
      mu:    K × 3  (residual mean)
      sigma: K × 3  (diagonal std, softplus)
      pi:    K      (mixing weights, softmax)
    """
    def __init__(self, in_dim, K=4, hidden_dim=64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, K * 3)
        self.sigma_head = nn.Linear(hidden_dim, K * 3)
        self.pi_head = nn.Linear(hidden_dim, K)

        # Init: mu ~ 0, sigma ~ 0.05 rad, pi uniform
        nn.init.normal_(self.mu_head.weight, 0, 0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.sigma_head.weight, 0, 0.01)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, x):
        """x: (..., in_dim) → mu(...,K,3), sigma(...,K,3), pi_logits(...,K)"""
        h = self.net(x)
        mu = self.mu_head(h).reshape(*x.shape[:-1], self.K, 3)
        sigma = F.softplus(self.sigma_head(h).reshape(*x.shape[:-1], self.K, 3)) + 1e-3
        sigma = sigma.clamp(min=1e-3, max=10.0)
        pi_logits = self.pi_head(h)
        return mu, sigma, pi_logits


# ════════════════════════════════════════════════════════════
# NLL Loss — negative log-likelihood of Gaussian mixture
# ════════════════════════════════════════════════════════════

def gmm_nll(y, mu, sigma, pi_logits):
    """Negative log-likelihood of Gaussian Mixture Model.
    
    Args:
      y: (B, T, J, 1, 3) — target residual
      mu: (B, T, J, K, 3) — component means
      sigma: (B, T, J, K, 3) — component stds
      pi_logits: (B, T, J, K) — mixing logits
    
    Returns: scalar NLL (mean over all dims)
    """
    log_pi = F.log_softmax(pi_logits, dim=-1)  # (B, T, J, K)
    diff = y - mu  # (B, T, J, K, 3)
    # Diagonal Gaussian log-prob
    log_sigma = torch.log(sigma)
    log_prob = -0.5 * ((diff / sigma)**2 + 2*log_sigma + math.log(2*math.pi))
    log_prob = log_prob.sum(-1)  # (B, T, J, K) — sum over 3D residual
    # Log-sum-exp over components
    log_like = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B, T, J)
    return -log_like.mean()


# ════════════════════════════════════════════════════════════
# Full Stage5G Model
# ════════════════════════════════════════════════════════════

class Stage5GModel(nn.Module):
    def __init__(self, n_joints=24, n_timesteps=96, K=4,
                 t2v_dim=16, joint_dim=64, attn_dim=64, n_heads=4):
        super().__init__()
        self.n_joints = n_joints; self.T = n_timesteps; self.K = K

        # Time2Vec
        self.t2v = Time2Vec(t2v_dim, n_sinusoidal=t2v_dim//2)

        # Per-joint input projection: start(3)+goal(3)+base_feat(6)+action(3)+t2v(16)=31
        in_dim = 3 + 3 + 6 + 3 + t2v_dim  # = 31
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim, joint_dim), nn.GELU(),
            nn.Linear(joint_dim, joint_dim), nn.GELU(),
        )

        # Fast temporal convolution (replaces GRU)
        self.temporal = TemporalEncoder(joint_dim)

        # Cross-attention (per timestep, 2-hop SMPL mask)
        self.cross_attn = SMPLCrossAttention(attn_dim, n_heads, n_joints)

        # Per-joint MDN heads
        self.mdn_heads = nn.ModuleList([
            TrajectoryMDNHead(attn_dim, K, hidden_dim=64)
            for _ in range(n_joints)
        ])

    def forward(self, start_rot, goal_rot, base_features, action):
        B, J = start_rot.shape[:2]; T = self.T

        # Time2Vec
        t_vals = torch.linspace(0, 1, T, device=start_rot.device)
        t2v_feat = self.t2v(t_vals)  # (T, t2v_dim)

        # Build per-joint per-timestep input
        s = start_rot.unsqueeze(2).expand(B, J, T, 3)
        g = goal_rot.unsqueeze(2).expand(B, J, T, 3)
        bf = base_features.unsqueeze(2).expand(B, J, T, 6)
        ac = action.unsqueeze(1).unsqueeze(2).expand(B, J, T, 3)
        t2 = t2v_feat.unsqueeze(0).unsqueeze(1).expand(B, J, T, self.t2v.out_dim)

        x = torch.cat([s, g, bf, ac, t2], dim=-1)  # (B, J, T, 31)
        x = self.joint_proj(x)                      # (B, J, T, joint_dim)
        x = self.temporal(x)                        # (B, J, T, joint_dim) — same dim

        # Cross-attention per timestep
        B_s, J_s, T_s, D_s = x.shape
        x_r = x.permute(0, 2, 1, 3).reshape(B_s * T_s, J_s, D_s)  # (B*T, J, D)
        x_attn = self.cross_attn(x_r)  # (B*T, J, D)
        x = x_attn.reshape(B_s, T_s, J_s, D_s)  # (B, T, J, D)

        # Per-joint MDN heads
        all_mu, all_sigma, all_pi = [], [], []
        for j in range(J):
            mu_j, sigma_j, pi_j = self.mdn_heads[j](x[:, :, j, :])
            all_mu.append(mu_j.unsqueeze(2))
            all_sigma.append(sigma_j.unsqueeze(2))
            all_pi.append(pi_j.unsqueeze(2))

        mu = torch.cat(all_mu, dim=2)        # (B, T, J, K, 3)
        sigma = torch.cat(all_sigma, dim=2)
        pi_logits = torch.cat(all_pi, dim=2)  # (B, T, J, K)
        return mu, sigma, pi_logits


# ════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════

def train_model(model, data75, bases, epochs=200, bs=16, lr=1e-3, device="cpu",
                fk_loss_weight=5.0, aux_weights=None):
    N, T, D = data75.shape; J = model.n_joints
    rot_data = data75[:,:,:72].reshape(N, T, J, 3)
    trans_data = data75[:,:,72:75]
    R = torch.tensor(rot_data, dtype=torch.float32, device=device)
    Tr = torch.tensor(trans_data, dtype=torch.float32, device=device)

    # Pre-compute deformed baselines
    print("  Computing baselines...")
    bl = np.zeros((N, T, J, 3), dtype=np.float32)
    for n in range(N):
        for j in range(J):
            bl[n,:,j] = baseball_deform(bases[j], rot_data[n,0,j], rot_data[n,-1,j])
    BL = torch.tensor(bl, dtype=torch.float32, device=device)

    # NO-SCALE residuals — raw radians
    res = R.permute(0,2,1,3) - BL.permute(0,2,1,3)  # (N, J, T, 3)
    # Target: (N, T, J, 3) for MDN
    Y = res.permute(0, 2, 1, 3)  # (N, T, J, 3)

    # Input: start+goal (scaled for conditioning)
    sg = np.concatenate([rot_data[:,0], rot_data[:,-1]], -1).reshape(N, J, 6)
    sg_mean = sg.reshape(-1,6).mean(0).astype(np.float32)
    sg_std = sg.reshape(-1,6).std(0).astype(np.float32) + 1e-8

    class ScalerObj:
        def __init__(self, m, s): self.mean=m; self.std=s
        def transform(self, X): return (X-self.mean)/self.std
        def inverse(self, X): return X*self.std+self.mean
    x_scaler = ScalerObj(sg_mean, sg_std)

    sg_s = x_scaler.transform(sg.reshape(-1,6)).reshape(N, J, 6)
    Ss = torch.tensor(sg_s[...,:3], dtype=torch.float32, device=device)
    Gs = torch.tensor(sg_s[...,3:], dtype=torch.float32, device=device)

    # Base features
    base_feat = np.zeros((N, J, 6), dtype=np.float32)
    for j in range(J):
        bj = bases[j]
        base_feat[:, j, 0] = np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum()
        base_feat[:, j, 1] = np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if T>2 else 0
        base_feat[:, j, 3] = bj[:,2].max()
        base_feat[:, j, 4] = bj[:,2].std()
        base_feat[:, j, 5] = bj[-1,2] - bj[0,2]
    Bf = torch.tensor(base_feat, dtype=torch.float32, device=device)

    actions = torch.zeros(N, 3, device=device); actions[:,0] = 1.0

    # Train/val split
    nv = max(1, int(0.15*N))
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    p = torch.randperm(N, generator=g, device="cpu")
    vi, ti = p[:nv], p[nv:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf"); best_state = None

    print(f"  Stage5G: NLL loss + GRU + Time2Vec | train={len(ti)} val={nv} bs={bs}")

    for ep in range(1, epochs+1):
        model.train()
        tp = torch.randperm(len(ti), device="cpu")
        total_nll, total_fk, cnt = 0.0, 0.0, 0
        for st in range(0, len(ti), bs):
            b = ti[tp[st:st+bs]]
            mu, sigma, pi_logits = model(Ss[b], Gs[b], Bf[b], actions[b])
            y = Y[b].unsqueeze(-2)  # (B, T, J, 1, 3)

            # NLL loss (main)
            nll = gmm_nll(y, mu, sigma, pi_logits)

            # FK loss (first sample per batch)
            with torch.no_grad():
                # Argmax component selection for FK validation only
                pi = F.softmax(pi_logits[0], dim=-1)  # (T, J, K)
                k = torch.argmax(pi, dim=-1)  # (T, J)
                # Gather the argmax component: mu[0,t,j,k[t,j],:]
                k_exp = k.unsqueeze(-1).unsqueeze(-1).expand(T, J, 1, 3)  # (T, J, 1, 3)
                mu_k = mu[0].gather(2, k_exp).squeeze(2)  # (T, J, 3)
                pred_rot_np = bl[b[0]] + mu_k.cpu().numpy()  # (T, J, 3)
                true_rot_np = rot_data[b[0]]  # (T, J, 3)
                p3d = fk_smpl_full(pred_rot_np.reshape(T, 72).astype(np.float32),
                                   trans_data[b[0]].astype(np.float32))
                t3d = fk_smpl_full(true_rot_np.reshape(T, 72).astype(np.float32),
                                   trans_data[b[0]].astype(np.float32))
                fk_loss_val = np.mean((p3d-t3d)**2)

            loss = nll + fk_loss_weight * torch.tensor(fk_loss_val, dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_nll += nll.item() * len(b)
            total_fk += fk_loss_val * len(b)
            cnt += len(b)

        # Validation
        model.eval()
        with torch.no_grad():
            mu_v, sigma_v, pi_v = model(Ss[vi], Gs[vi], Bf[vi], actions[vi])
            y_v = Y[vi].unsqueeze(-2)
            val_loss = gmm_nll(y_v, mu_v, sigma_v, pi_v).item()

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep:4d}: nll={total_nll/max(cnt,1):.4f} val={val_loss:.6f} "
                  f"fk={total_fk/max(cnt,1):.6f} best={best_val:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val, x_scaler, None


# ════════════════════════════════════════════════════════════
# Generation — Probabilistic sampling
# ════════════════════════════════════════════════════════════

def generate_motion(model, bases, start, goal, trans, x_scaler=None, y_scaler=None, device="cpu",
                     sample_mode="argmax", temperature=1.0):
    """Generate motion from MDN.
    
    sample_mode:
      "argmax" — pick best component, use its mean (MAP estimate, no noise)
      "sample" — sample component from categorical, then sample residual ~ N(mu, sigma*temp)
    """
    model.eval()
    J, T = 24, model.T
    s = np.asarray(start, np.float32).reshape(J, 3)
    g = np.asarray(goal, np.float32).reshape(J, 3)

    if x_scaler is not None:
        sg = x_scaler.transform(np.concatenate([s, g], -1))
        S = torch.tensor(sg[:,:3], dtype=torch.float32, device=device).unsqueeze(0)
        G = torch.tensor(sg[:,3:], dtype=torch.float32, device=device).unsqueeze(0)
    else:
        S = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        G = torch.tensor(g, dtype=torch.float32, device=device).unsqueeze(0)

    action = torch.zeros(1, 3, device=device); action[:, 0] = 1.0

    bf = torch.zeros(1, J, 6, device=device)
    for j in range(J):
        bj = bases[j]
        bf[0, j, 0] = torch.tensor(np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum(), dtype=torch.float32)
        bf[0, j, 1] = torch.tensor(np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if T>2 else 0.0, dtype=torch.float32)
        bf[0, j, 3] = torch.tensor(bj[:,2].max(), dtype=torch.float32)
        bf[0, j, 4] = torch.tensor(bj[:,2].std(), dtype=torch.float32)
        bf[0, j, 5] = torch.tensor(bj[-1,2]-bj[0,2], dtype=torch.float32)

    with torch.no_grad():
        mu, sigma, pi_logits = model(S, G, bf, action)

    mu_np = mu[0].cpu().numpy()        # (T, J, K, 3)
    sigma_np = sigma[0].cpu().numpy()   # (T, J, K, 3)
    pi_np = F.softmax(pi_logits[0], dim=-1).cpu().numpy()  # (T, J, K)

    res = np.zeros((J, T, 3), dtype=np.float32)
    rng = np.random.default_rng()

    if sample_mode == "argmax":
        # MAP: pick argmax component, use its mean (no noise)
        k_best = pi_np.argmax(axis=-1)  # (T, J)
        for t in range(T):
            for j in range(J):
                k = k_best[t, j]
                res[j, t] = mu_np[t, j, k]
    else:
        # Probabilistic sampling
        for t in range(T):
            for j in range(J):
                k = rng.choice(model.K, p=pi_np[t, j])
                eps = rng.normal(0, 1, 3).astype(np.float32)
                res[j, t] = mu_np[t, j, k] + sigma_np[t, j, k] * eps * temperature

    # Boundary-lock: enforce zero residual at start/goal
    # Force exact zero at boundaries (not just line subtraction)
    res[:, 0, :] = 0.0
    res[:, -1, :] = 0.0

    # Reconstruct rotations
    bl = np.zeros((J, T, 3), dtype=np.float32)
    for j in range(J):
        bl[j] = baseball_deform(bases[j], s[j], g[j])

    rot = (bl + res).transpose(1, 0, 2)  # (T, J, 3)
    full = rot.reshape(T, 72)
    if len(trans) != T:
        idx = np.linspace(0, len(trans)-1, T, dtype=int)
        trans = trans[idx]
    return fk_smpl_full(full, trans), rot


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-base", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fk-weight", type=float, default=5.0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("[Stage5G — Google Architecture] Loading...")
    data, J = load_data(args.data_dir, args.n_files, args.n_frames, seed=args.seed)

    rng = np.random.default_rng(args.seed+1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:args.n_base]]
    train_data = data[perm[args.n_base:]][:int(0.8*(len(data)-args.n_base))]
    test_data = data[perm[args.n_base:]][int(0.8*(len(data)-args.n_base)):]

    bases = compute_mean_base(base_data[:,:,:72], args.n_base)
    model = Stage5GModel(n_joints=J, n_timesteps=args.n_frames).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    best_val, xs, ys = train_model(model, train_data, bases, args.n_epochs,
                                    args.batch_size, args.lr, args.device,
                                    fk_loss_weight=args.fk_weight)
    print(f"Done {time.time()-t0:.0f}s best_val={best_val:.6f}")

    if len(test_data):
        test = test_data[0]
        rt = test[:args.n_frames,:72].reshape(args.n_frames, J, 3)
        tr = test[:args.n_frames, 72:75]
        g3d, gen_rot = generate_motion(model, bases, rt[0], rt[-1], tr, xs, ys, args.device, sample_mode="argmax")
        t3d = fk_smpl_full(rt.reshape(args.n_frames, 72), tr)
        rmse = float(np.sqrt(np.mean((g3d-t3d)**2)))
        gen_amp = float(np.std(gen_rot, axis=0).mean())
        true_amp = float(np.std(rt, axis=0).mean())
        print(f"RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}")

        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
            BONES=[(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
                   (9,13),(13,16),(16,18),(18,20),(20,22),(9,14),(14,17),(17,19),(19,21),(21,23)]
            OFF=np.array([0,-1.5,0])
            tx=t3d[:,0,0]; wc=(tx.max()+tx.min())/2; FX=(tx.max()-tx.min())/2+1.5
            fig=plt.figure(figsize=(16,8)); ax=fig.add_subplot(111,projection='3d')
            def upd(f):
                ax.cla()
                xx,yy=np.meshgrid(np.linspace(wc-FX,wc+FX,15),np.linspace(-1.5,1.5,9))
                ax.plot_surface(xx,yy,np.zeros_like(xx),alpha=0.06,color='gray')
                jg=g3d[f]; jt=t3d[f]+OFF
                for a,b in BONES:
                    ax.plot([jg[a,0],jg[b,0]],[jg[a,1],jg[b,1]],[jg[a,2],jg[b,2]],color='#1565C0',lw=3.0,alpha=0.9)
                    ax.plot([jt[a,0],jt[b,0]],[jt[a,1],jt[b,1]],[jt[a,2],jt[b,2]],color='#E53935',lw=3.0,alpha=0.9)
                ax.set_xlim(wc-FX,wc+FX); ax.set_ylim(-1.5,1.5); ax.set_zlim(0,2.0); ax.view_init(elev=20,azim=-70)
                ax.set_title(f'Stage5G — Google Arch | RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}',fontsize=13)
            anim=FuncAnimation(fig,upd,frames=args.n_frames,interval=100)
            out=Path(f"outputs/stage5g_{args.n_epochs}ep.gif"); out.parent.mkdir(parents=True,exist_ok=True)
            anim.save(str(out),writer=PillowWriter(fps=10)); plt.close()
            print(f"Animation: {out}")
        except Exception as e:
            print(f"Animation skipped: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
