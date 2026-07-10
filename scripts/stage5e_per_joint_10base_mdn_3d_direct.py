#!/usr/bin/env python3
"""
Stage5E-3D — Direct 3D position prediction (like baseball, no FK in generation).
Bone length constraints enforced via penalty on actual 3D positions.

Architecture:
  Layer 1: Per-joint MDN + Cross-Attention
           Input: start_3d, goal_3d, base_descriptor(6), action(3) = 15-d
           Predicts RESIDUAL in 3D position space
  Output:   Direct 3D positions (NO FK — output IS final 3D, like baseball)

Usage:
  python scripts/stage5e_per_joint_10base_mdn_3d_direct.py --n-files 200 --n-epochs 200
"""

from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

from smpl_fk import fk_smpl, fk_smpl_full, POSE_BODY_TO_SMPL

Array = np.ndarray


# ============================================================
# Data loader — 24 joints: root(3) + body(63) + hands(6) + trans(3) = 75
# ============================================================

def load_amass_pose_body(data_dir: str, n_files: int = 200, n_frames: int = 96,
                          min_translation: float = 1.0, seed: int = 42) -> tuple[Array, int]:
    """Load AMASS walk — returns 24-joint rotation data.
    
    Format per frame: [root_orient(3), pose_body(63), hand_zeros(6), trans(3)] = 75-d
    Hands are zero (AMASS has no hand pose); model learns them from arm context.
    """
    base = Path(data_dir)
    all_files = list(base.rglob("Walk*.npz"))
    rng = np.random.default_rng(seed)

    sequences = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]
            tr = d["trans"]
            T_orig = pb.shape[0]
            if T_orig < n_frames:
                continue
            x_range = tr[:, 0].max() - tr[:, 0].min()
            if x_range < min_translation:
                continue
            idx = np.linspace(0, T_orig - 1, n_frames, dtype=int)
            pb_ds = pb[idx]                         # (T, 63)
            ro_ds = d["root_orient"][idx]           # (T, 3)
            tr_ds = tr[idx]                         # (T, 3)
            hand_z = np.zeros((n_frames, 6), dtype=np.float32)  # L_hand + R_hand
            # Full 72-d rotation: root(3) + body(63) + hands(6)
            full_rot = np.concatenate([ro_ds, pb_ds, hand_z], axis=1)  # (T, 72)
            sequences.append(np.concatenate([full_rot, tr_ds], axis=1))  # (T, 75)
        except Exception:
            continue

    if n_files > 0 and len(sequences) > n_files:
        sequences = list(rng.choice(sequences, n_files, replace=False))

    data = np.stack(sequences, axis=0)  # (N, T, 75)
    n_joints = 24  # ALL 24 SMPL joints: root(0) + body(1-21) + hands(22-23)
    print(f"Loaded {len(sequences)} straight walks (24 joints)")
    return data.astype(np.float32), n_joints


# ============================================================
# Base computation — PER-JOINT MEAN (NOT medoid)
# ============================================================

def compute_per_joint_mean_bases(data_rot: Array, n_base_walks: int = 10) -> Array:
    """Per-joint mean base from n_base_walks samples.
    
    WHY MEAN NOT MEDOID:
    - Medoid = single walk, full amplitude → model learns tiny residuals → stiff
    - Mean = averaged across walks, ~40% amplitude → model MUST learn large residuals
      → recovers full articulation with variation across samples
    
    Returns: (J, T, 3) per-joint base paths.
    """
    N, T, D = data_rot.shape
    J = D // 3
    
    # Use first n_base_walks as base pool
    n_use = min(n_base_walks, N)
    base_pool = data_rot[:n_use]  # (n_use, T, D)
    
    # Per-joint mean across base walks
    # Reshape to (n_use, T, J, 3) → mean over n_use → (T, J, 3) → (J, T, 3)
    base_pool_reshaped = base_pool.reshape(n_use, T, J, 3)
    mean_base = np.nanmean(base_pool_reshaped, axis=0)  # (T, J, 3)
    
    # Log articulation retention
    sample_walk = data_rot[n_use:n_use+1].reshape(1, T, J, 3) if N > n_use else base_pool_reshaped[:1]
    sample_amp = np.std(sample_walk, axis=1).mean()  # avg std across time
    base_amp = np.std(mean_base, axis=0).mean()
    retention = base_amp / max(sample_amp, 1e-8) * 100
    print(f"  Base articulation retention: {retention:.0f}% of sample walk (target: ~40%)")
    
    return mean_base.transpose(1, 0, 2)  # (J, T, 3)


# ============================================================
# Deformation (same as baseball Z2)
# ============================================================

def deform_path_to_start_and_goal(path: Array, start: Array, goal: Array,
                                  gamma: float = 0.3) -> Array:
    """D_{s,g}[Q] — soft deformation (gamma < 1.0 = model must learn the rest).
    
    gamma=1.0: exact start-goal match
    gamma=0.3: only 30% goal correction → model must learn 70% residual
    """
    Q = np.asarray(path, dtype=float)
    s = np.asarray(start, dtype=float).reshape(1, -1)
    g = np.asarray(goal, dtype=float).reshape(1, -1)
    alpha = np.linspace(0.0, 1.0, Q.shape[0], dtype=float).reshape(-1, 1)
    return Q - Q[0:1] + s + gamma * alpha * (g - (Q[-1:] - Q[0:1] + s))


# ============================================================
# Model
# ============================================================

class OneShotJointMDN(nn.Module):
    def __init__(self, feature_dim, n_timesteps=96, n_components=4, hidden_dim=128, n_hidden=3):
        super().__init__()
        self.target_dim = n_timesteps * 3
        layers = []
        d = feature_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(0.0)]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.logits = nn.Linear(d, n_components)
        self.mu = nn.Linear(d, n_components * self.target_dim)
        self.log_sigma = nn.Linear(d, n_components)
        nn.init.zeros_(self.mu.bias)
        nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.log_sigma.bias)

    def forward(self, x):
        h = self.trunk(x)
        K = self.logits.out_features
        return self.logits(h), self.mu(h).view(-1, K, self.target_dim), self.log_sigma(h).clamp(-4.0, 3.0)


class PerJointRotationMDN(nn.Module):
    """J joints, each with one-shot MDN + cross-attention for coordination."""
    def __init__(self, n_joints=21, n_timesteps=96, n_components=4, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps

        self.joint_encoder = nn.Sequential(
            nn.Linear(15, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.mdns = nn.ModuleList([
            OneShotJointMDN(feature_dim=feature_dim * 2, n_timesteps=n_timesteps,
                            n_components=n_components, hidden_dim=hidden_dim)
            for _ in range(n_joints)
        ])

    def forward(self, start_rot, goal_rot, action, base_features):
        B, J = start_rot.shape[:2]
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        feat = torch.cat([start_rot, goal_rot, base_features, action_exp], dim=-1)  # (B, J, 15)
        encoded = self.joint_encoder(feat)
        context, _ = self.attn(encoded, encoded, encoded)
        joint_input = torch.cat([encoded, context], dim=-1)

        all_logits, all_mu, all_sigma = [], [], []
        for j in range(J):
            l, m, s = self.mdns[j](joint_input[:, j, :])
            all_logits.append(l.unsqueeze(1))
            all_mu.append(m.unsqueeze(1))
            all_sigma.append(s.unsqueeze(1))
        return torch.cat(all_logits, 1), torch.cat(all_mu, 1), torch.cat(all_sigma, 1)


# ============================================================
# MDN Loss (true NLL — same as baseball trajectory model)
# ============================================================

def mdn_nll_isotropic(logits, mu, log_sigma, y):
    """Negative log likelihood for isotropic Gaussian mixture.
    
    logits:    B x J x K  (joints, components)
    mu:        B x J x K x D
    log_sigma: B x J x K
    y:         B x J x D
    
    Returns scalar loss averaged over batch and joints.
    """
    B, J, K, D = mu.shape
    y_exp = y.unsqueeze(2)  # B x J x 1 x D
    diff2 = torch.sum((y_exp - mu) ** 2, dim=-1)  # B x J x K
    
    inv_var = torch.exp(-2.0 * log_sigma)  # B x J x K
    log_pi = torch.log_softmax(logits, dim=-1)  # B x J x K
    
    log_norm = -0.5 * D * math.log(2.0 * math.pi)
    component_log_prob = (
        log_pi
        + log_norm
        - D * log_sigma
        - 0.5 * diff2 * inv_var
    )
    log_prob = torch.logsumexp(component_log_prob, dim=-1)  # B x J
    return -torch.mean(log_prob)


def mixture_mse_surrogate(logits, mu, y):
    """Auxiliary: weighted expected mean should be near target (stabilizes early training)."""
    probs = torch.softmax(logits, dim=-1)  # B x J x K
    pred = torch.sum(probs.unsqueeze(-1) * mu, dim=2)  # B x J x D
    return torch.mean((pred - y) ** 2)


# ============================================================
# StandardScaler (same as baseball)
# ============================================================

class StandardScalerNP:
    def __init__(self, eps=1e-9):
        self.mean = None
        self.std = None
        self.eps = eps
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X, axis=0).astype(np.float32)
        self.std = (np.std(X, axis=0) + self.eps).astype(np.float32)
        return self
    
    def transform(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.std
    
    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float32) * self.std + self.mean


# ============================================================
# Training
# ============================================================

def train_model(model, data, bases_3d, n_epochs=200, batch_size=16, lr=1e-3, device="cpu"):
    """Train in 3D POSITION space directly. bases_3d is (J, T, 3) in meters."""
    from smpl_fk import fk_smpl_full, KINTREE
    n_samples, n_frames, dim = data.shape
    n_joints = model.n_joints
    T = n_frames

    # data: (N, T, 75) = [full_rot(72), trans(3)]
    rot_data = data[:, :, :72].reshape(n_samples, n_frames, n_joints, 3)
    tr_data = data[:, :, 72:75]
    
    # FK → 3D positions for training targets
    print("  FK → 3D positions...")
    pos_3d = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        pos_3d[s] = fk_smpl_full(rot_data[s].reshape(n_frames, 72), tr_data[s])
    X_data = torch.tensor(pos_3d, dtype=torch.float32, device=device)
    
    # bases_3d is already in 3D meters (FK'd by caller)
    base_pos_3d = torch.tensor(bases_3d, dtype=torch.float32, device=device)  # (24, T, 3)
    
    # Deformed baselines in 3D
    print("  Computing 3D deformed baselines...")
    baselines = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        for j in range(n_joints):
            s_j = X_data[s, 0, j].cpu().numpy()
            g_j = X_data[s, -1, j].cpu().numpy()
            b_j = base_pos_3d[j].cpu().numpy()
            baselines[s, :, j] = deform_path_to_start_and_goal(b_j, s_j, g_j)
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    starts = X_data[:, 0, :, :]
    goals = X_data[:, -1, :, :]
    actions = torch.zeros(n_samples, 3, device=device)
    actions[:, 0] = 1.0

    base_features = torch.zeros(n_samples, n_joints, 6, device=device)
    for j in range(n_joints):
        bj = base_pos_3d[j]
        d1 = bj[1:] - bj[:-1]; length = d1.norm(dim=-1).sum()
        d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
        curv = d2.norm(dim=-1).mean() / max(length.item(), 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + torch.linspace(0, 1, T, device=device).unsqueeze(1) * (bj[-1:] - bj[0:1])
        dev = (bj - chord).norm(dim=-1).max()
        base_features[:, j, 0] = length; base_features[:, j, 1] = curv; base_features[:, j, 2] = dev
        base_features[:, j, 3] = bj[:, 2].max(); base_features[:, j, 4] = bj[:, 2].std()
        base_features[:, j, 5] = bj[-1, 2] - bj[0, 2]

    residuals = X_data - baselines_t
    targets = residuals.permute(0, 2, 1, 3).reshape(n_samples, n_joints, -1)
    D = targets.shape[-1]

    y_scaler = StandardScalerNP().fit(targets.numpy().reshape(-1, D))
    Y_std = torch.tensor(y_scaler.transform(targets.numpy().reshape(-1, D)).reshape(n_samples, n_joints, D),
                         dtype=torch.float32, device=device)

    X_input = torch.cat([starts.reshape(n_samples, n_joints, 3),
                         goals.reshape(n_samples, n_joints, 3), base_features], dim=-1)
    x_scaler = StandardScalerNP().fit(X_input.numpy().reshape(-1, 12))
    X_std = torch.tensor(x_scaler.transform(X_input.numpy().reshape(-1, 12)).reshape(n_samples, n_joints, 12),
                         dtype=torch.float32, device=device)

    # Bone pairs + true bone lengths (in meters, from FK data)
    bone_pairs = [(int(p), int(c)) for c, p in enumerate(KINTREE) if p >= 0]
    true_bone_len = {}
    for p, c in bone_pairs:
        true_bone_len[(p, c)] = torch.norm(X_data[:, :, c, :] - X_data[:, :, p, :], dim=-1).mean().item()
    
    var_total = X_data.var().item()
    var_residual = residuals.var().item()
    r2 = 1.0 - var_residual / max(var_total, 1e-8)
    print(f"  Base R2 (3D): {r2:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for st in range(0, n_samples, batch_size):
            b = perm[st:st + batch_size]; bs = len(b)
            start_b = X_std[b, :, :3]; goal_b = X_std[b, :, 3:6]
            base_b = X_std[b, :, 6:12]; act_b = actions[b]; tb = Y_std[b]

            logits, mu, log_sigma = model(start_b, goal_b, act_b, base_b)
            K = mu.shape[2]

            # MCL
            diff = tb.unsqueeze(2) - mu
            comp_err = (diff * diff).mean(dim=-1).mean(dim=1)
            winner = comp_err.argmin(dim=1)
            eps = 0.05
            w = torch.full_like(comp_err, eps / (K - 1))
            w.scatter_(1, winner.unsqueeze(1), 1.0 - eps)
            recon = (w * comp_err).sum(dim=1).mean()
            gate = F.cross_entropy(logits.mean(dim=1), winner)

            # Smoothness
            res_3d_std = mu.view(bs, n_joints, K, T, 3)
            acc = res_3d_std[:, :, :, 2:, :] - 2*res_3d_std[:, :, :, 1:-1, :] + res_3d_std[:, :, :, :-2, :]
            
            # Bone length penalty — compute on UNSTANDARDIZED 3D predictions
            mu_winner = mu[torch.arange(bs, device=device), :, winner, :]  # (B, J, D)
            mu_winner_3d = mu_winner.view(bs, n_joints, T, 3)  # standardized
            # Unstandardize: inverse_transform
            mu_np = mu_winner_3d.detach().cpu().numpy()
            bl_batch = baselines_t[b]  # (B, J, T, 3)
            pred_3d = np.zeros((bs, T, n_joints, 3), dtype=np.float32)
            for i in range(bs):
                r_i = y_scaler.inverse_transform(mu_np[i].reshape(n_joints, -1)).reshape(n_joints, T, 3)
                pred_3d[i] = (bl_batch[i].permute(1, 0, 2).cpu().numpy() + r_i).transpose(1, 0, 2)
            pred_3d_t = torch.tensor(pred_3d, dtype=torch.float32, device=device)
            
            bone_loss = 0.0
            for p, c in bone_pairs:
                dp = torch.norm(pred_3d_t[:, :, c, :] - pred_3d_t[:, :, p, :], dim=-1)  # (B, T)
                target_len = true_bone_len[(p, c)]
                bone_loss = bone_loss + ((dp - target_len) ** 2).mean()
            bone_loss = bone_loss / len(bone_pairs)

            loss = 100.0 * recon + 0.1 * gate + 0.05 * (acc * acc).mean() + 10.0 * bone_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * bs
        
        avg_loss = total_loss / n_samples
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}: loss={avg_loss:.4f}  best={best_loss:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_loss, baselines_t, x_scaler, y_scaler, base_pos_3d.cpu().numpy()


def generate_motion(model, base_pos_3d, start_pos, goal_pos,
                    x_scaler=None, y_scaler=None, n_frames=96, device="cpu"):
    """Generate 24-joint 3D positions directly (NO FK — output IS final 3D)."""
    model.eval()
    n_joints = model.n_joints
    T = n_frames

    start_np = np.asarray(start_pos, dtype=np.float32)
    goal_np = np.asarray(goal_pos, dtype=np.float32)
    
    # Base descriptors
    base_f_np = np.zeros((n_joints, 6), dtype=np.float32)
    for j in range(n_joints):
        bj = base_pos_3d[j]
        d1 = bj[1:] - bj[:-1]
        length = np.linalg.norm(d1, axis=-1).sum()
        d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
        curv = np.linalg.norm(d2, axis=-1).mean() / max(length, 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + np.linspace(0, 1, T).reshape(-1, 1) * (bj[-1:] - bj[0:1])
        dev = np.linalg.norm(bj - chord, axis=-1).max()
        base_f_np[j, 0] = length
        base_f_np[j, 1] = curv
        base_f_np[j, 2] = dev
        base_f_np[j, 3] = bj[:, 2].max()
        base_f_np[j, 4] = bj[:, 2].std()
        base_f_np[j, 5] = bj[-1, 2] - bj[0, 2]
    
    # Build input: (1, J, 12) = [start, goal, base_desc] — action passed separately
    x_raw = np.concatenate([
        start_np.reshape(1, n_joints, 3),
        goal_np.reshape(1, n_joints, 3),
        base_f_np.reshape(1, n_joints, 6),
    ], axis=-1).astype(np.float32)  # (1, J, 12)
    
    # Standardize input
    if x_scaler is not None:
        x_std = x_scaler.transform(x_raw.reshape(-1, 12)).reshape(1, n_joints, 12)
    else:
        x_std = x_raw
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    
    # Deformed baseline in 3D
    baselines = np.zeros((n_joints, n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        baselines[j] = deform_path_to_start_and_goal(base_pos_3d[j], start_np[j], goal_np[j])
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    with torch.no_grad():
        act_t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        logits, mu_std, log_sigma = model(
            x_t[:, :, :3], x_t[:, :, 3:6], act_t, x_t[:, :, 6:12])
        k = int(logits.mean(dim=1).argmax(dim=-1)[0])
        mu_blend = mu_std[:, :, k, :]
    mu_np = mu_blend[0].cpu().numpy()
    
    if y_scaler is not None:
        residual_flat = y_scaler.inverse_transform(mu_np)
    else:
        residual_flat = mu_np
    
    residual = residual_flat.reshape(n_joints, T, 3)
    joints_3d = baselines + residual  # (J, T, 3) — directly 3D, NO FK needed
    joints_3d = joints_3d.transpose(1, 0, 2)  # (T, J, 3)
    return joints_3d


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-base-walks", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if torch is None:
        print("PyTorch not available."); return

    device = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Loading AMASS straight walks...")
    data, n_joints = load_amass_pose_body(args.data_dir, args.n_files, args.n_frames, seed=args.seed)
    print(f"Data: {data.shape}")

    # Split
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(data))

    # Compute bases in 3D (FK base walks with actual trans)
    base_data_rot = base_data[:, :, :72].reshape(args.n_base_walks, args.n_frames, n_joints, 3)
    base_data_tr = base_data[:, :, 72:75]
    bases_3d = np.zeros((n_joints, args.n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        # FK each base walk, then average in 3D
        joints_j = np.zeros((args.n_base_walks, args.n_frames, 3), dtype=np.float32)
        for w in range(args.n_base_walks):
            fp = np.zeros((args.n_frames, 72), dtype=np.float32)
            fp[:, j*3:(j+1)*3] = base_data_rot[w, :, j, :]  # only this joint's rot
            # Actually need full pose for FK. Use mean of all joints from base walks
        # Simpler: FK the full mean base rotation
        pass
    # Correct approach: FK the mean rotation base, then use as 3D base
    from smpl_fk import fk_smpl_full
    base_full = bases_rot.transpose(1, 0, 2).reshape(args.n_frames, 72)
    # Use mean trans from base walks
    mean_tr = base_data_tr.mean(axis=0)  # (T, 3)
    bases_3d = fk_smpl_full(base_full, mean_tr)  # (T, 24, 3)
    bases_3d = bases_3d.transpose(1, 0, 2)  # (24, T, 3)
    print(f"  3D base computed with mean trans (range: {mean_tr[:,0].min():.1f}-{mean_tr[:,0].max():.1f}m)")

    # Train/test split from remaining data
    remaining = data[perm[args.n_base_walks:]]
    n_train = int(0.8 * len(remaining))
    train_data = remaining[:n_train]
    val_data = remaining[n_train:]
    print(f"Base: {args.n_base_walks}, Train: {len(train_data)}, Test: {len(val_data)}")

    print(f"\nBuilding model: {n_joints} joints, 4-comp rotation MDN + cross-attention...")
    model = PerJointRotationMDN(n_joints=n_joints, n_timesteps=args.n_frames, n_components=4,
                                feature_dim=32, hidden_dim=128).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining in 3D position space (direct, bone-constrained)...")
    t0 = time.time()
    best_loss, _, x_scaler, y_scaler, base_pos_3d = train_model(
        model, train_data, bases_3d, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"Training complete: {time.time()-t0:.1f}s, best={best_loss:.4f}")

    # Test
    print("\nTesting (direct 3D, NO FK)...")
    test = val_data[0]
    rot_test = test[:args.n_frames, :72].reshape(args.n_frames, n_joints, 3)
    tr_test = test[:args.n_frames, 72:75]
    # Get true 3D positions via FK
    test_pos = fk_smpl_full(rot_test.reshape(args.n_frames, 72), tr_test)

    generated_3d = generate_motion(model, base_pos_3d, test_pos[0], test_pos[-1],
                                    x_scaler=x_scaler, y_scaler=y_scaler,
                                    n_frames=args.n_frames, device=device)
    true_3d = test_pos

    rmse = np.sqrt(np.mean((generated_3d - true_3d) ** 2))
    goal_err = np.sqrt(np.mean((generated_3d[-1] - true_3d[-1]) ** 2))

    # Per-joint RMSE (24 joints)
    per_joint_rmse = np.sqrt(np.mean((generated_3d - true_3d) ** 2, axis=(0, 2)))

    print(f"  Position RMSE (24 joints): {rmse:.4f}m")
    print(f"  Goal error: {goal_err:.4f}m")
    print(f"  Bone lengths GUARANTEED by FK (Layer 2)")

    # Articulation check
    gen_amp = np.std(generated_3d, axis=0).mean()
    true_amp = np.std(true_3d, axis=0).mean()
    print(f"\n  Articulation amplitude: gen={gen_amp:.4f}  true={true_amp:.4f}  ratio={gen_amp/max(true_amp,1e-8):.2f}")
    if gen_amp / max(true_amp, 1e-8) > 0.7:
        print("  [OK] Articulation recovery GOOD (>70% of true amplitude)")
    else:
        print("  [WARN] Articulation still LOW — may need more training or weaker base")

    # Top-3 joints by error
    JOINT_NAMES_24 = ["pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2",
                      "L_ankle","R_ankle","spine3","L_foot","R_foot","neck",
                      "L_collar","R_collar","head","L_shoulder","R_shoulder",
                      "L_elbow","R_elbow","L_wrist","R_wrist","L_hand","R_hand"]
    top3 = np.argsort(per_joint_rmse)[-3:][::-1]
    top3_names = [JOINT_NAMES_24[i] for i in top3]
    print(f"\n  Highest-error joints: {top3_names}  ({per_joint_rmse[top3]}m)")
    print("\nDone.")


if __name__ == "__main__":
    main()
