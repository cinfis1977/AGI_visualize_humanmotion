#!/usr/bin/env python3
"""
Stage5E — Per-Joint Rotation MDN + Cross-Attention.

Architecture (faithful to baseball trajectory model):
  Layer 1: Per-joint Rotation MDN + Cross-Attention
           Input: start_rot(3) + goal_rot(3) + base_desc(6) + action(3) = 15-d
           Predicts RESIDUAL from deformed mean-base
  Layer 2: FK → 3D positions (bone lengths GUARANTEED)

Training: MCL (winner-take-most) + StandardScaler + AdamW
Generation: argmax component → FK → final 3D

Usage:
  python scripts/stage5e_per_joint_10base_mdn.py --n-files 200 --n-epochs 200
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

from smpl_fk import fk_smpl_full

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
                                  gamma: float = 1.0) -> Array:
    """D_{s,g}[Q] — full deformation (gamma=1.0 = model predicts articulation, not goal correction)."""
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
    """J joints, each with one-shot MDN + phase-aware cross-attention + phase skip + smoother."""
    def __init__(self, n_joints=21, n_timesteps=96, n_components=4, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps

        # Encoder: geometry(15) + phase(8) = 23-d. Phase goes to BOTH encoder AND MDN skip.
        self.joint_encoder = nn.Sequential(
            nn.Linear(23, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        # MDN input: encoded(32) + context(32) + phase_skip(8) = 72-d
        self.mdns = nn.ModuleList([
            OneShotJointMDN(feature_dim=feature_dim * 2 + 8, n_timesteps=n_timesteps,
                            n_components=n_components, hidden_dim=hidden_dim)
            for _ in range(n_joints)
        ])
        smooth_in = n_joints * 12
        self.smooth = nn.Conv1d(smooth_in, smooth_in, kernel_size=3,
                                padding=1, groups=n_joints, bias=False)
        nn.init.dirac_(self.smooth.weight)

    def forward(self, start_rot, goal_rot, action, base_features, apply_smooth=False):
        B, J = start_rot.shape[:2]
        K = self.mdns[0].logits.out_features
        T = self.n_timesteps
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        # base_features: (B, J, 14) = shape(6) + phase(8)
        base_shape = base_features[:, :, :6]    # geometry
        base_phase = base_features[:, :, 6:14]  # timing
        # Encoder: geometry + phase → attention CAN see timing for coordination
        feat_all = torch.cat([start_rot, goal_rot, base_shape, base_phase, action_exp], dim=-1)  # 23-d
        encoded = self.joint_encoder(feat_all)   # (B, J, 32) — now phase-aware
        context, _ = self.attn(encoded, encoded, encoded)  # (B, J, 32) — time-dependent attention
        # MDN: encoded + context + phase skip = double phase signal
        joint_input = torch.cat([encoded, context, base_phase], dim=-1)  # (B, J, 72)

        all_logits, all_mu, all_sigma = [], [], []
        for j in range(J):
            l, m, s = self.mdns[j](joint_input[:, j, :])
            all_logits.append(l.unsqueeze(1))
            all_mu.append(m.unsqueeze(1))
            all_sigma.append(s.unsqueeze(1))
        logits = torch.cat(all_logits, 1)
        mu = torch.cat(all_mu, 1)
        sigma = torch.cat(all_sigma, 1)
        
        if apply_smooth:
            # Per-joint Conv1d: (B, J*K*3, T) with groups=J
            Bs = mu.shape[0]
            mu_r = mu.view(Bs, J, K, T, 3).permute(0, 1, 2, 4, 3).reshape(Bs, J * K * 3, T)
            mu_smooth = self.smooth(mu_r)
            mu_smooth = mu_smooth.view(Bs, J, K, 3, T).permute(0, 1, 2, 4, 3).reshape(Bs, J, K, -1)
            return logits, mu_smooth, sigma
        
        return logits, mu, sigma


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

def _compute_base_features(bases_rot, n_joints, T, device="cpu"):
    """Compute base path features (shape + rotation-based phase) for all joints.
    
    Returns (n_joints, 14): 6 shape features + 8 phase features.
    Phase features = sin/cos of when rotation angle (not 3D position) reaches extremes.
    """
    feats = np.zeros((n_joints, 14), dtype=np.float32)
    for j in range(n_joints):
        bj = bases_rot[j]  # (T, 3) — axis-angle rotation vectors
        # -- Shape features (6) --
        d1 = bj[1:] - bj[:-1]; length = np.linalg.norm(d1, axis=-1).sum()
        d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
        curv = np.linalg.norm(d2, axis=-1).mean() / max(length, 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + np.linspace(0, 1, T).reshape(-1, 1) * (bj[-1:] - bj[0:1])
        dev = np.linalg.norm(bj - chord, axis=-1).max()
        feats[j, 0] = length; feats[j, 1] = curv; feats[j, 2] = dev
        feats[j, 3] = bj[:, 2].max(); feats[j, 4] = bj[:, 2].std()
        feats[j, 5] = bj[-1, 2] - bj[0, 2]
        # -- Phase features (8): sin/cos from ROTATION angle extremes --
        rot_angle = np.linalg.norm(bj, axis=1)          # (T,) — rotation magnitude over time
        ph_max_rot = np.argmax(rot_angle) / T
        ph_min_rot = np.argmin(rot_angle) / T
        ph_max_z   = np.argmax(bj[:, 2]) / T             # vertical rotation component
        ph_min_z   = np.argmin(bj[:, 2]) / T
        feats[j, 6] = np.sin(2*np.pi*ph_max_rot); feats[j, 7] = np.cos(2*np.pi*ph_max_rot)
        feats[j, 8] = np.sin(2*np.pi*ph_min_rot); feats[j, 9] = np.cos(2*np.pi*ph_min_rot)
        feats[j,10] = np.sin(2*np.pi*ph_max_z);   feats[j,11] = np.cos(2*np.pi*ph_max_z)
        feats[j,12] = np.sin(2*np.pi*ph_min_z);   feats[j,13] = np.cos(2*np.pi*ph_min_z)
    return feats

def train_model(model, data, bases_rot, n_epochs=200, batch_size=16, lr=1e-3, device="cpu"):
    """Train in ROTATION space with MCL + standardization + validation."""
    n_samples, n_frames, dim = data.shape
    n_joints = model.n_joints
    T = n_frames

    # data: (N, T, 75) = [full_rot(72), trans(3)]
    rot_data = data[:, :, :72].reshape(n_samples, n_frames, n_joints, 3)
    X_data = torch.tensor(rot_data, dtype=torch.float32, device=device)
    
    # Deformed baselines in ROTATION space
    print("  Computing deformed baselines (rotation space)...")
    baselines = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        for j in range(n_joints):
            s_j = X_data[s, 0, j].cpu().numpy()
            g_j = X_data[s, -1, j].cpu().numpy()
            b_j = bases_rot[j]
            baselines[s, :, j] = deform_path_to_start_and_goal(b_j, s_j, g_j)
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    starts = X_data[:, 0, :, :]
    goals = X_data[:, -1, :, :]
    actions = torch.zeros(n_samples, 3, device=device)
    actions[:, 0] = 1.0

    base_features_np = _compute_base_features(bases_rot, n_joints, T)
    base_features = torch.tensor(base_features_np, dtype=torch.float32, device=device).unsqueeze(0).expand(n_samples, -1, -1)

    residuals = X_data - baselines_t
    targets = residuals.permute(0, 2, 1, 3).reshape(n_samples, n_joints, -1)
    D = targets.shape[-1]

    y_scaler = StandardScalerNP().fit(targets.numpy().reshape(-1, D))
    Y_std = torch.tensor(y_scaler.transform(targets.numpy().reshape(-1, D)).reshape(n_samples, n_joints, D),
                         dtype=torch.float32, device=device)

    X_input = torch.cat([starts.reshape(n_samples, n_joints, 3),
                         goals.reshape(n_samples, n_joints, 3), base_features], dim=-1)
    # CRITICAL: base_features (shape+phase = 14-d) are CONSTANT across samples
    # (derived from base path only). StandardScaler would divide by ~0 std → explosion.
    # Only standardize start+goal (6 dims that vary per sample). Base features stay raw.
    GEO_DIM = 6  # start(3) + goal(3) ONLY
    x_scaler = StandardScalerNP().fit(X_input.numpy().reshape(-1, 20)[:, :GEO_DIM])
    geo_std = x_scaler.transform(X_input.numpy().reshape(-1, 20)[:, :GEO_DIM])
    base_raw = X_input.numpy().reshape(-1, 20)[:, GEO_DIM:]  # (N*J, 14) — raw shape+phase
    X_std_np = np.concatenate([geo_std, base_raw], axis=-1).reshape(n_samples, n_joints, 20)
    X_std = torch.tensor(X_std_np, dtype=torch.float32, device=device)

    # Split 15% validation (deterministic — same split every run)
    n_val = max(1, int(0.15 * n_samples))
    g = torch.Generator(device=device)
    g.manual_seed(42)
    shuffled = torch.randperm(n_samples, generator=g, device=device)
    val_idx = shuffled[:n_val]
    tr_idx = shuffled[n_val:]
    X_tr, Y_tr = X_std[tr_idx], Y_std[tr_idx]
    X_val, Y_val = X_std[val_idx], Y_std[val_idx]
    val_act = actions[val_idx]

    var_total = X_data.var().item()
    var_residual = residuals.var().item()
    r2 = 1.0 - var_residual / max(var_total, 1e-8)
    print(f"  Base R2: {r2:.3f}  train={len(tr_idx)} val={len(val_idx)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_idx))
        total_loss = 0.0
        for st in range(0, len(tr_idx), batch_size):
            b = perm[st:st + batch_size]; bs = len(b)
            b_idx = tr_idx[b]
            start_b = X_tr[b, :, :3]; goal_b = X_tr[b, :, 3:6]
            base_b = X_tr[b, :, 6:20]; act_b = actions[b_idx]; tb = Y_tr[b]

            logits, mu, log_sigma = model(start_b, goal_b, act_b, base_b, apply_smooth=True)
            K = mu.shape[2]

            diff = tb.unsqueeze(2) - mu                     # (B, J, K, D)
            comp_err = (diff * diff).mean(dim=-1)          # (B, J, K) — per-joint error
            winner = comp_err.argmin(dim=-1)                # (B, J) — per-joint winner
            eps = 0.05
            w = torch.full_like(comp_err, eps / (K - 1))    # (B, J, K)
            w.scatter_(-1, winner.unsqueeze(-1), 1.0 - eps)
            recon = (w * comp_err).sum(dim=-1).mean()       # avg over (B,J,K)
            gate = sum(F.cross_entropy(logits[:, j, :], winner[:, j])
                      for j in range(model.n_joints)) / model.n_joints

            loss = 100.0 * recon + 0.1 * gate

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * bs
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, v_mu, v_sigma = model(
                X_val[:, :, :3], X_val[:, :, 3:6], val_act, X_val[:, :, 6:20], apply_smooth=True)
            v_diff = Y_val.unsqueeze(2) - v_mu
            v_err = (v_diff * v_diff).mean(dim=-1)          # (B, J, K) — per-joint
            v_winner = v_err.argmin(dim=-1)                 # (B, J)
            v_eps = 0.05
            v_w = torch.full_like(v_err, v_eps / (K - 1))
            v_w.scatter_(-1, v_winner.unsqueeze(-1), 1.0 - v_eps)
            val_loss = (v_w * v_err).sum(dim=-1).mean().item()
        
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}: train={total_loss/len(tr_idx):.4f}  val={val_loss:.4f}  best_val={best_val_loss:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss, baselines_t, x_scaler, y_scaler


def generate_motion(model, bases_rot, start_rot, goal_rot, trans,
                    x_scaler=None, y_scaler=None, n_frames=96, device="cpu"):
    """Generate 24-joint rotation -> FK -> 3D positions."""
    model.eval()
    n_joints = model.n_joints
    T = n_frames

    start_np = np.asarray(start_rot, dtype=np.float32)
    goal_np = np.asarray(goal_rot, dtype=np.float32)
    
    base_f_np = _compute_base_features(bases_rot, n_joints, T)
    
    x_raw = np.concatenate([start_np.reshape(1, n_joints, 3),
                            goal_np.reshape(1, n_joints, 3),
                            base_f_np.reshape(1, n_joints, 14)], axis=-1).astype(np.float32)
    
    GEO_DIM = 6  # start(3) + goal(3) ONLY — what scaler was fit on
    if x_scaler is not None:
        x_flat = x_raw.reshape(-1, 20)
        geo_std = x_scaler.transform(x_flat[:, :GEO_DIM])          # standardize start+goal
        base_raw = x_flat[:, GEO_DIM:]                              # keep base features raw
        x_std = np.concatenate([geo_std, base_raw], axis=-1).reshape(1, n_joints, 20)
    else:
        x_std = x_raw
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    
    baselines = np.zeros((n_joints, n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        baselines[j] = deform_path_to_start_and_goal(bases_rot[j], start_rot[j], goal_rot[j])

    with torch.no_grad():
        act_t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        logits, mu_std, log_sigma = model(x_t[:, :, :3], x_t[:, :, 3:6], act_t, x_t[:, :, 6:20], apply_smooth=True)
        # Per-joint component selection: each joint picks its own best component
        k_per_joint = logits[0].argmax(dim=-1)              # (J,) — best component per joint
        mu_blend = torch.stack([mu_std[0, j, k_per_joint[j], :] for j in range(n_joints)]).unsqueeze(0)
    mu_np = mu_blend[0].cpu().numpy()
    
    if y_scaler is not None:
        residual_flat = y_scaler.inverse_transform(mu_np)
    else:
        residual_flat = mu_np
    
    residual = residual_flat.reshape(n_joints, T, 3)
    pb_np = baselines + residual
    pb_np = pb_np.transpose(1, 0, 2)

    full_pose = pb_np.reshape(T, 72)
    if len(trans) != T:
        idx = np.linspace(0, len(trans) - 1, T, dtype=int)
        trans = trans[idx]
    joints_3d = fk_smpl_full(full_pose, trans)
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

    # Base: use n_base_walks samples — extract 72-d rotation from data[:,:,:72]
    base_data = data[perm[:args.n_base_walks]]
    rot_base = base_data[:, :, :72].reshape(args.n_base_walks, args.n_frames, n_joints, 3)

    print(f"\nComputing per-joint MEAN bases from {args.n_base_walks} walks...")
    bases_rot = compute_per_joint_mean_bases(
        rot_base.reshape(args.n_base_walks, args.n_frames, -1),
        n_base_walks=args.n_base_walks
    )

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

    print(f"\nTraining in rotation space (MEAN base -> large residuals, 24 joints)...")
    t0 = time.time()
    train_rot = train_data[:, :, :72].reshape(len(train_data), args.n_frames, -1)
    best_loss, _, x_scaler, y_scaler = train_model(
        model, train_rot, bases_rot, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"Training complete: {time.time()-t0:.1f}s, best_val={best_loss:.4f}")

    # Test
    print("\nTesting (rotation -> FK -> 3D, ALL 24 joints)...")
    test = val_data[0]
    rot_test = test[:args.n_frames, :72].reshape(args.n_frames, n_joints, 3)  # (T, 24, 3)
    tr_test = test[:args.n_frames, 72:75]  # (T, 3)
    start_rot = rot_test[0]   # (24, 3)
    goal_rot = rot_test[-1]   # (24, 3)

    generated_3d = generate_motion(model, bases_rot, start_rot, goal_rot,
                                   tr_test, x_scaler=x_scaler, y_scaler=y_scaler,
                                   n_frames=args.n_frames, device=device)
    # True: FK on original data
    true_3d = fk_smpl_full(rot_test.reshape(args.n_frames, 72), tr_test)  # (T, 24, 3)

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
