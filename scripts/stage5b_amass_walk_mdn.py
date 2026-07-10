#!/usr/bin/env python3
"""
Stage5B — Real AMASS Walk Data with Per-Joint MDN + Cross-Attention.

Architecture (faithful to Z2/baseball):
  1. Load N AMASS walk files, downsample to 32 frames
  2. Extract pose_body (63-d) → reshape to (21 joints × 3 rot params)
  3. Per-joint base paths: Q_j = mean(train_trajectories[:,:,j])
  4. D_{s,g}[Q_j] deformation to match start_j, goal_j
  5. Per-joint MDN predicts RESIDUAL from baseline
  6. Cross-attention coordinates joints

Usage:
  python scripts/stage5b_amass_walk_mdn.py --n-files 50 --n-epochs 200
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

Array = np.ndarray


# ============================================================
# AMASS data loader
# ============================================================

def load_amass_walk_files(data_dir: str, n_files: int = 100, n_frames: int = 32,
                          seed: int = 42) -> tuple[Array, int]:
    """Load AMASS walk files, downsample, convert to 3D joint positions via FK.
    Returns: (N, T, J*3) array in meters, n_joints=21 (body joints only)
    """
    from smpl_fk import fk_smpl, POSE_BODY_TO_SMPL

    base = Path(data_dir)
    walk_files = sorted(base.rglob("Walk*.npz"))
    rng = np.random.default_rng(seed)
    if n_files > 0 and len(walk_files) > n_files:
        walk_files = list(rng.choice(walk_files, n_files, replace=False))

    n_joints = 21  # SMPL body joints (1-21)
    dim = n_joints * 3  # 63

    sequences = []
    for f in walk_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]
            ro = d["root_orient"]
            tr = d["trans"]
            T_orig = pb.shape[0]
            if T_orig < n_frames:
                continue
            # Downsample to n_frames
            idx = np.linspace(0, T_orig - 1, n_frames, dtype=int)
            pb_ds = pb[idx]
            ro_ds = ro[idx]
            tr_ds = tr[idx]

            # Convert to 3D joint positions
            joints_3d = fk_smpl(pb_ds, ro_ds, tr_ds)  # (n_frames, 24, 3)
            # Take body joints only (indices 1-21 in SMPL = 21 joints)
            body_joints = joints_3d[:, POSE_BODY_TO_SMPL, :]  # (n_frames, 21, 3)
            sequences.append(body_joints.reshape(n_frames, -1))
        except Exception as e:
            continue

    data = np.stack(sequences, axis=0)
    print(f"Loaded {len(sequences)} walk sequences ({n_joints} joints x 3D in meters)")
    return data.astype(np.float32), n_joints


# ============================================================
# Deformation (same as Z2)
# ============================================================

def deform_path_to_start_and_goal(path: Array, start: Array, goal: Array) -> Array:
    Q = np.asarray(path, dtype=float)
    s = np.asarray(start, dtype=float).reshape(1, -1)
    g = np.asarray(goal, dtype=float).reshape(1, -1)
    alpha = np.linspace(0.0, 1.0, Q.shape[0], dtype=float).reshape(-1, 1)
    return Q - Q[0:1] + s + alpha * (g - (Q[-1:] - Q[0:1] + s))


def compute_bases(data: Array) -> Array:
    """Per-joint base paths: mean over training samples. Returns (J, T, 3)."""
    N, T, D = data.shape
    J = D // 3
    shaped = data.reshape(N, T, J, 3)
    mean_paths = np.nanmean(shaped, axis=0)  # (T, J, 3)
    return mean_paths.transpose(1, 0, 2)  # → (J, T, 3)


# ============================================================
# One-Shot Model (faithful to Z2/baseball)
# ============================================================

class OneShotJointMDN(nn.Module):
    """One-shot MDN: feature vector → T×3 residual for ONE joint.
    Same architecture as Z2 ResidualPathMDN, just smaller target dim (T×3).
    """
    def __init__(self, feature_dim, n_timesteps=32, n_components=4,
                 hidden_dim=128, n_hidden=2, dropout=0.0):
        super().__init__()
        self.target_dim = n_timesteps * 3  # flattened T×3
        layers = []
        d = feature_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
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
        logits = self.logits(h)
        mu = self.mu(h).view(-1, K, self.target_dim)
        log_sigma = self.log_sigma(h).clamp(-4.0, 3.0)
        return logits, mu, log_sigma


class PerJointOneShotMDN(nn.Module):
    """J joints, each with its own one-shot MDN + cross-attention coordinator."""
    def __init__(self, n_joints=21, n_timesteps=32, n_components=4,
                 feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps
        self.target_dim = n_timesteps * 3

        # Per-joint feature encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(15, feature_dim), nn.GELU(),  # [start(3), goal(3), base_desc(6), action(3)]
            nn.Linear(feature_dim, feature_dim),
        )

        # Cross-attention (on joint feature space)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

        # Per-joint one-shot MDNs
        self.mdns = nn.ModuleList([
            OneShotJointMDN(feature_dim=feature_dim * 2, n_timesteps=n_timesteps,
                            n_components=n_components, hidden_dim=hidden_dim)
            for _ in range(n_joints)
        ])

    def forward(self, start_pose, goal_pose, base_features, action):
        """
        start_pose:    (B, J, 3)
        goal_pose:     (B, J, 3)
        base_features: (B, J, 6)  — path descriptors per joint
        action:        (B, 3)
        Returns: per-joint (logits, mu, log_sigma)
        """
        B, J = start_pose.shape[:2]
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        feat = torch.cat([start_pose, goal_pose, base_features, action_exp], dim=-1)  # (B, J, 15)
        encoded = self.joint_encoder(feat)  # (B, J, F)

        # Cross-attention
        context, _ = self.attn(encoded, encoded, encoded)  # (B, J, F)

        # Per-joint MDN: concatenate self + context
        joint_input = torch.cat([encoded, context], dim=-1)  # (B, J, 2F)

        all_logits, all_mu, all_sigma = [], [], []
        for j in range(J):
            l, m, s = self.mdns[j](joint_input[:, j, :])
            all_logits.append(l.unsqueeze(1))
            all_mu.append(m.unsqueeze(1))
            all_sigma.append(s.unsqueeze(1))
        return torch.cat(all_logits, 1), torch.cat(all_mu, 1), torch.cat(all_sigma, 1)


# ============================================================
# Loss (one-shot)
# ============================================================

def mdn_nll_one_shot(logits, mu, log_sigma, target_flat):
    """NLL for one-shot MDN.
    logits:     (B, J, K)
    mu:         (B, J, K, T*3)
    log_sigma:  (B, J, K)
    target_flat: (B, J, T*3) — ground truth residual
    """
    B, J, K = logits.shape
    D = mu.shape[-1]  # T*3
    diff = target_flat.unsqueeze(2) - mu  # (B, J, K, D)
    inv_var = torch.exp(-2.0 * log_sigma).unsqueeze(-1)
    log_prob = -0.5 * (diff * diff * inv_var).sum(dim=-1) - D * log_sigma - 0.5 * D * math.log(2 * math.pi)
    log_prob_joint = log_prob.sum(dim=1)  # (B, K)
    log_pi_avg = F.log_softmax(logits.mean(dim=1), dim=-1)
    return -torch.logsumexp(log_pi_avg + log_prob_joint, dim=-1).mean()


# ============================================================
# Training (one-shot)
# ============================================================

def train_model_one_shot(model, data, bases, n_epochs=200, batch_size=16, lr=1e-3, device="cpu"):
    """One-shot training: feature vector → flattened residual (T×3 per joint)."""
    n_samples, n_frames, dim = data.shape
    n_joints = model.n_joints
    X_data = torch.tensor(data, dtype=torch.float32, device=device).view(n_samples, n_frames, n_joints, 3)
    T = n_frames

    # Pre-compute deformed baselines
    print("  Computing deformed baselines...")
    baselines = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        for j in range(n_joints):
            s_j = X_data[s, 0, j].cpu().numpy()
            g_j = X_data[s, -1, j].cpu().numpy()
            b_j = bases[j]
            baselines[s, :, j] = deform_path_to_start_and_goal(b_j, s_j, g_j)
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    # Start poses, goal poses per sample
    starts = X_data[:, 0, :, :]   # (N, J, 3)
    goals = X_data[:, -1, :, :]   # (N, J, 3)

    # Base descriptors per joint (path length, curvature proxy)
    base_features = torch.zeros(n_samples, n_joints, 6, device=device)
    for j in range(n_joints):
        bj = torch.tensor(bases[j], dtype=torch.float32, device=device)  # (T, 3)
        d1 = bj[1:] - bj[:-1]
        length = d1.norm(dim=-1).sum()
        d2 = bj[2:] - 2 * bj[1:-1] + bj[:-2]
        curv = d2.norm(dim=-1).mean() / max(length.item(), 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + torch.linspace(0, 1, T, device=device).unsqueeze(1) * (bj[-1:] - bj[0:1])
        dev = (bj - chord).norm(dim=-1).max()
        z_max = bj[:, 2].max()
        base_features[:, j, 0] = length
        base_features[:, j, 1] = curv
        base_features[:, j, 2] = dev
        base_features[:, j, 3] = z_max
        base_features[:, j, 4] = bj[:, 2].std()
        base_features[:, j, 5] = (bj[-1, 2] - bj[0, 2])

    # Action encoding
    actions = torch.zeros(n_samples, 3, device=device)
    actions[:, 0] = 1.0

    # Targets: flattened residuals from baseline
    residuals = X_data - baselines_t  # (N, T, J, 3)
    targets = residuals.permute(0, 2, 1, 3).reshape(n_samples, n_joints, -1)  # (N, J, T*3)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for st in range(0, n_samples, batch_size):
            b = perm[st:st + batch_size]
            logits, mu, log_sigma = model(starts[b], goals[b], base_features[b], actions[b])
            loss = mdn_nll_one_shot(logits, mu, log_sigma, targets[b])

            # Acceleration penalty (zigzag killer — same as Z2)
            # Applied to weighted mean residual, flattened back to (B, J, T, 3)
            probs = F.softmax(logits, dim=-1)
            weighted_res = (probs.unsqueeze(-1) * mu).sum(dim=2)  # (B, J, T*3)
            res_3d = weighted_res.view(len(b), n_joints, T, 3)  # (B, J, T, 3)
            # Second diff along time
            acc = res_3d[:, :, 2:, :] - 2*res_3d[:, :, 1:-1, :] + res_3d[:, :, :-2, :]
            acc_pen = (acc * acc).mean()
            loss = loss + 0.05 * acc_pen  # smoothness weight 0.05

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * len(b)
        avg_loss = total_loss / n_samples
        if avg_loss < best_loss:
            best_loss = avg_loss
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}: loss={avg_loss:.4f}  best={best_loss:.4f}")
    return best_loss, baselines_t


def generate_motion_one_shot(model, bases, start_pose, goal_pose, n_frames=32, device="cpu"):
    """One-shot generation from start to goal."""
    model.eval()
    n_joints = model.n_joints
    T = n_frames

    start_t = torch.tensor(start_pose, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal_pose, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)

    # Pre-compute baselines
    baselines = np.zeros((n_joints, n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        baselines[j] = deform_path_to_start_and_goal(bases[j], start_pose[j], goal_pose[j])
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    # Base features — computed from RAW base (same as training), not deformed baseline
    base_f = torch.zeros(1, n_joints, 6, device=device)
    for j in range(n_joints):
        bj = torch.tensor(bases[j], dtype=torch.float32, device=device)  # (T, 3) — raw base
        d1 = bj[1:] - bj[:-1]
        length = d1.norm(dim=-1).sum()
        d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
        curv = d2.norm(dim=-1).mean() / max(length.item(), 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + torch.linspace(0, 1, T, device=device).unsqueeze(1) * (bj[-1:] - bj[0:1])
        dev = (bj - chord).norm(dim=-1).max()
        z_max = bj[:, 2].max()
        base_f[0, j, 0] = length
        base_f[0, j, 1] = curv
        base_f[0, j, 2] = dev
        base_f[0, j, 3] = z_max
        base_f[0, j, 4] = bj[:, 2].std()
        base_f[0, j, 5] = bj[-1, 2] - bj[0, 2]

    with torch.no_grad():
        logits, mu, log_sigma = model(start_t, goal_t, base_f, action_t)
        probs = F.softmax(logits, dim=-1)  # (1, J, K)
        residual_flat = (probs.unsqueeze(-1) * mu).sum(dim=2)  # (1, J, T*3)
        residual = residual_flat.view(1, n_joints, n_frames, 3)  # (1, J, T, 3)
        generated = baselines_t.unsqueeze(0) + residual  # (1, J, T, 3)
        generated = generated[0].permute(1, 0, 2)  # (T, J, 3)
        generated_np = generated.cpu().numpy()

        # Savitzky-Golay smoothing (zigzag killer — same as Z2)
        from scipy.signal import savgol_filter
        w = min(7, n_frames - 2 if (n_frames - 2) % 2 == 1 else n_frames - 3)
        if w >= 5:
            for j in range(n_joints):
                for d in range(3):
                    generated_np[:, j, d] = savgol_filter(generated_np[:, j, d], w, 3)

    return generated_np


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--n-files", type=int, default=50)
    p.add_argument("--base-samples", type=int, default=0, help="Samples for base only (0=all train); tests few-shot base learning")
    p.add_argument("--n-frames", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if torch is None:
        print("PyTorch not available.")
        return

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading AMASS walk data...")
    data_flat, n_joints = load_amass_walk_files(args.data_dir, args.n_files, args.n_frames, args.seed)
    print(f"Data: {data_flat.shape}  ({n_joints} joints x 3D = {n_joints*3})")

    # Split: base_set (few samples for base only) + train_set + test_set
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(data_flat))
    n_base = args.base_samples if args.base_samples > 0 else 0

    if n_base > 0:
        base_data = data_flat[perm[:n_base]]
        remaining = data_flat[perm[n_base:]]
        n_train = int(0.8 * len(remaining))
        train_data = remaining[:n_train]
        val_data = remaining[n_train:]
        print(f"Base samples: {n_base}, Train: {len(train_data)}, Test: {len(val_data)}")
    else:
        n_train = int(0.8 * len(data_flat))
        train_data = data_flat[:n_train]
        val_data = data_flat[n_train:]
        base_data = train_data  # base from all training data

    bases = compute_bases(base_data)
    print(f"Bases: {bases.shape} (from {len(base_data)} samples)")

    print(f"\nBuilding model: {n_joints} joints, one-shot 4-comp MDN + cross-attention...")
    model = PerJointOneShotMDN(n_joints=n_joints, n_timesteps=args.n_frames, n_components=4,
                               feature_dim=32, hidden_dim=128).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining on {n_train} samples (one-shot)...")
    t0 = time.time()
    best_loss, baselines = train_model_one_shot(model, train_data, bases, n_epochs=args.n_epochs,
                                                batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"Training complete: {time.time()-t0:.1f}s, best_loss={best_loss:.4f}")

    print("\nGenerating test motion (one-shot)...")
    test_sample = val_data[0].reshape(args.n_frames, n_joints, 3)
    start = test_sample[0]
    goal = test_sample[-1]
    generated = generate_motion_one_shot(model, bases, start, goal, n_frames=args.n_frames, device=device)

    true_rmse = np.sqrt(np.mean((test_sample - generated) ** 2))
    goal_err = np.sqrt(np.mean((generated[-1] - goal) ** 2))
    print(f"  True vs generated RMSE: {true_rmse:.4f}")
    print(f"  Goal error: {goal_err:.4f}")

    print("\nPer-joint RMSE:")
    for j in range(min(n_joints, 10)):
        j_rmse = np.sqrt(np.mean((test_sample[:, j, :] - generated[:, j, :]) ** 2))
        print(f"  joint {j:2d}: {j_rmse:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
