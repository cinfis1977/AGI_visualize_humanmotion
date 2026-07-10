#!/usr/bin/env python3
"""
Stage5D — Multi-Base Articulated Human Motion MDN.

FIXES the stiffness problem from Stage5C:
  - Stage5C used ONE medoid base → all generated walks had same articulation amplitude
  - Stage5D clusters walks by articulation style → 3-5 prototype bases
  - Model conditions on base_id → can generate narrow/medium/wide step walks

Architecture (two-layer, faithful to Z2/baseball):
  Layer 1: Per-joint Rotation MDN + Cross-Attention + Base Selector
  Layer 2: FK (bone lengths GUARANTEED)

Key differences from Stage5C:
  - compute_bases() → compute_multi_bases() (k-medoids clustering by articulation)
  - Input includes base_id (one-hot) so model knows which articulation style to use
  - Articulation metrics: step_length, knee_angle_range, ankle_height_range

Usage:
  python scripts/stage5d_articulated_base_mdn.py --n-files 200 --n-clusters 4 --n-epochs 200
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

from smpl_fk import fk_smpl, POSE_BODY_TO_SMPL

Array = np.ndarray


# ============================================================
# Data loader (same as Stage5C)
# ============================================================

def load_amass_pose_body(data_dir: str, n_files: int = 200, n_frames: int = 96,
                          min_translation: float = 1.0, seed: int = 42) -> tuple[Array, int]:
    """Load AMASS walk pose_body directly (rotations, not positions)."""
    base = Path(data_dir)
    all_files = list(base.rglob("Walk*.npz"))
    rng = np.random.default_rng(seed)

    sequences = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]  # (T_orig, 63)
            tr = d["trans"]
            T_orig = pb.shape[0]
            if T_orig < n_frames:
                continue
            x_range = tr[:, 0].max() - tr[:, 0].min()
            if x_range < min_translation:
                continue
            idx = np.linspace(0, T_orig - 1, n_frames, dtype=int)
            pb_ds = pb[idx]
            ro_ds = d["root_orient"][idx]
            tr_ds = tr[idx]
            sequences.append(np.concatenate([pb_ds, ro_ds, tr_ds], axis=1))
        except Exception:
            continue

    if n_files > 0 and len(sequences) > n_files:
        sequences = list(rng.choice(sequences, n_files, replace=False))

    data = np.stack(sequences, axis=0)  # (N, n_frames, 69)
    n_joints = 21
    print(f"Loaded {len(sequences)} straight walks")
    return data.astype(np.float32), n_joints


# ============================================================
# Articulation metrics for clustering
# ============================================================

def compute_articulation_features(data_rot: Array) -> Array:
    """Compute per-walk articulation features for clustering.
    
    Returns (N, 4): [step_length, knee_flex_range, ankle_z_range, hip_abd_range]
    All computed in rotation space (axis-angle norms as proxy for joint angles).
    """
    N, T, D = data_rot.shape
    J = D // 3
    data_j = data_rot.reshape(N, T, J, 3)  # (N, T, J, 3)

    features = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        # 1. Step length proxy: total angular displacement of hip joints over walk
        hip_rots = data_j[i, :, 1:3, :]  # joints 1,2 = hips, (T, 2, 3)
        step_len = np.linalg.norm(hip_rots[-1] - hip_rots[0], axis=-1).mean()

        # 2. Knee flexion range: max-min rotation magnitude at knees (joints 4,5)
        knee_rots = data_j[i, :, 4:6, :]  # (T, 2, 3)
        knee_mag = np.linalg.norm(knee_rots, axis=-1)  # (T, 2)
        knee_range = (knee_mag.max(axis=0) - knee_mag.min(axis=0)).mean()

        # 3. Ankle z-range: elevation variation at ankles (joints 7,8)
        ankle_rots = data_j[i, :, 7:9, :]  # (T, 2, 3)
        ankle_mag = np.linalg.norm(ankle_rots, axis=-1)
        ankle_range = (ankle_mag.max(axis=0) - ankle_mag.min(axis=0)).mean()

        # 4. Hip abduction range: side-to-side at hips
        hip_abd = np.abs(data_j[i, :, 1:3, 0])  # x-component = abduction proxy
        hip_abd_range = (hip_abd.max(axis=0) - hip_abd.min(axis=0)).mean()

        features[i] = [step_len, knee_range, ankle_range, hip_abd_range]

    # Normalize
    f_mean = features.mean(axis=0, keepdims=True)
    f_std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - f_mean) / f_std


def compute_multi_bases(data_rot: Array, n_clusters: int = 4) -> tuple[Array, Array, Array]:
    """Cluster walks by articulation style, return one base per cluster.

    Returns:
        bases: (K, J, T, 3) — one base path per cluster
        cluster_ids: (N,) — cluster assignment for each walk
        artic_features: (N, 4) — normalized articulation features
    """
    N, T, D = data_rot.shape
    J = D // 3

    artic_feat = compute_articulation_features(data_rot)

    # Simple k-means using Euclidean distance on articulation features
    rng = np.random.default_rng(42)
    # Initialize with k-means++
    centers = [artic_feat[rng.integers(N)]]
    for _ in range(1, n_clusters):
        dists = np.min([np.linalg.norm(artic_feat - c, axis=1) for c in centers], axis=0)
        probs = dists / (dists.sum() + 1e-8)
        centers.append(artic_feat[rng.choice(N, p=probs)])
    centers = np.array(centers)

    # Lloyd's algorithm
    for _ in range(20):
        dists = np.array([np.linalg.norm(artic_feat - c, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        new_centers = np.array([artic_feat[labels == k].mean(axis=0) if (labels == k).any()
                                else centers[k] for k in range(n_clusters)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # For each cluster, find medoid walk as base
    bases = np.zeros((n_clusters, J, T, 3), dtype=np.float32)
    data_j = data_rot.reshape(N, T, J, 3)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            # Empty cluster: use global medoid
            mean_flat = np.nanmean(data_rot, axis=0)
            dist = np.linalg.norm(np.nan_to_num(data_rot - mean_flat[None]), axis=(1, 2))
            medoid = data_j[int(np.argmin(dist))]
        elif mask.sum() == 1:
            bases[k] = data_j[mask][0].transpose(1, 0, 2)
            continue
        else:
            cluster_data = data_rot[mask]
            mean_flat = np.nanmean(cluster_data, axis=0)
            dist = np.linalg.norm(np.nan_to_num(cluster_data - mean_flat[None]), axis=(1, 2))
            medoid = data_j[mask][int(np.argmin(dist))]
        bases[k] = medoid.transpose(1, 0, 2)  # (J, T, 3)

    # Print cluster stats
    for k in range(n_clusters):
        n_k = (labels == k).sum()
        c = centers[k]
        print(f"  Cluster {k}: {n_k} walks | step={c[0]:.2f} knee={c[1]:.2f} ankle={c[2]:.2f} hip={c[3]:.2f}")

    return bases, labels, artic_feat


# ============================================================
# Deformation (same as Stage5C)
# ============================================================

def deform_path_to_start_and_goal(path: Array, start: Array, goal: Array) -> Array:
    Q = np.asarray(path, dtype=float)
    s = np.asarray(start, dtype=float).reshape(1, -1)
    g = np.asarray(goal, dtype=float).reshape(1, -1)
    alpha = np.linspace(0.0, 1.0, Q.shape[0], dtype=float).reshape(-1, 1)
    return Q - Q[0:1] + s + alpha * (g - (Q[-1:] - Q[0:1] + s))


# ============================================================
# Model (extended with base selector)
# ============================================================

class OneShotJointMDN(nn.Module):
    def __init__(self, feature_dim, n_timesteps=96, n_components=4, hidden_dim=128, n_hidden=2):
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


class PerJointRotationMDN_v2(nn.Module):
    """J joints, each with one-shot MDN + cross-attention + base_id conditioning."""
    def __init__(self, n_joints=21, n_timesteps=96, n_components=4, n_bases=4,
                 feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps
        self.n_bases = n_bases

        # Input: start(3) + goal(3) + base_desc(6) + action(3) + base_id(n_bases) = 15 + n_bases
        input_dim = 15 + n_bases
        self.joint_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.mdns = nn.ModuleList([
            OneShotJointMDN(feature_dim=feature_dim * 2, n_timesteps=n_timesteps,
                            n_components=n_components, hidden_dim=hidden_dim)
            for _ in range(n_joints)
        ])

    def forward(self, start_rot, goal_rot, action, base_features, base_id):
        B, J = start_rot.shape[:2]
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        base_id_exp = base_id.unsqueeze(1).expand(B, J, self.n_bases)
        feat = torch.cat([start_rot, goal_rot, base_features, action_exp, base_id_exp], dim=-1)
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
# Loss (same MCL + smoothness as Stage5C)
# ============================================================

def train_model(model, data, multi_bases, cluster_ids, n_epochs=200, batch_size=16, lr=1e-3, device="cpu"):
    """Train in ROTATION space with multi-base conditioning."""
    n_samples, n_frames, dim = data.shape
    n_joints = model.n_joints
    K_bases = model.n_bases
    T = n_frames

    pb_data = data[:, :, :63].reshape(n_samples, n_frames, n_joints, 3)
    X_data = torch.tensor(pb_data, dtype=torch.float32, device=device)

    # Pre-compute deformed baselines for EACH sample using its assigned cluster base
    print("  Computing multi-base deformed baselines...")
    baselines = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        k = cluster_ids[s]
        for j in range(n_joints):
            s_j = X_data[s, 0, j].cpu().numpy()
            g_j = X_data[s, -1, j].cpu().numpy()
            b_j = multi_bases[k, j]
            baselines[s, :, j] = deform_path_to_start_and_goal(b_j, s_j, g_j)
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    starts = X_data[:, 0, :, :]
    goals = X_data[:, -1, :, :]
    actions = torch.zeros(n_samples, 3, device=device)
    actions[:, 0] = 1.0

    # Base ID (one-hot)
    base_id = torch.zeros(n_samples, K_bases, device=device)
    base_id[range(n_samples), cluster_ids] = 1.0

    # Base descriptors per joint (from cluster-specific base)
    base_features = torch.zeros(n_samples, n_joints, 6, device=device)
    for s in range(n_samples):
        k = cluster_ids[s]
        for j in range(n_joints):
            bj = torch.tensor(multi_bases[k, j], dtype=torch.float32, device=device)
            d1 = bj[1:] - bj[:-1]
            length = d1.norm(dim=-1).sum()
            d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
            curv = d2.norm(dim=-1).mean() / max(length.item(), 1e-8) if len(d2) > 0 else 0.0
            chord = bj[0:1] + torch.linspace(0, 1, T, device=device).unsqueeze(1) * (bj[-1:] - bj[0:1])
            dev = (bj - chord).norm(dim=-1).max()
            z_max = bj[:, 2].max()
            base_features[s, j, 0] = length
            base_features[s, j, 1] = curv
            base_features[s, j, 2] = dev
            base_features[s, j, 3] = z_max
            base_features[s, j, 4] = bj[:, 2].std()
            base_features[s, j, 5] = bj[-1, 2] - bj[0, 2]

    residuals = X_data - baselines_t
    targets = residuals.permute(0, 2, 1, 3).reshape(n_samples, n_joints, -1)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for st in range(0, n_samples, batch_size):
            b = perm[st:st + batch_size]
            logits, mu, log_sigma = model(starts[b], goals[b], actions[b],
                                           base_features[b], base_id[b])
            tb = targets[b]
            K = mu.shape[2]

            # eps-relaxed MCL
            diff = tb.unsqueeze(2) - mu
            comp_err = (diff * diff).mean(dim=-1).mean(dim=1)
            winner = comp_err.argmin(dim=1)
            eps = 0.05
            w = torch.full_like(comp_err, eps / (K - 1))
            w.scatter_(1, winner.unsqueeze(1), 1.0 - eps)
            recon = (w * comp_err).sum(dim=1).mean()

            gate = F.cross_entropy(logits.mean(dim=1), winner)

            res_3d = mu.view(len(b), n_joints, K, T, 3)
            acc = res_3d[:, :, :, 2:, :] - 2*res_3d[:, :, :, 1:-1, :] + res_3d[:, :, :, :-2, :]
            loss = 100.0 * recon + 0.1 * gate + 0.05 * (acc * acc).mean()

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


def generate_motion(model, multi_bases, base_id, start_rot, goal_rot, root_orient, trans,
                    n_frames=96, device="cpu"):
    """Generate using a specific base cluster."""
    model.eval()
    n_joints = model.n_joints
    K_bases = model.n_bases
    T = n_frames
    k = int(base_id)

    start_t = torch.tensor(start_rot, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal_rot, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)
    bid_t = torch.zeros(1, K_bases, device=device)
    bid_t[0, k] = 1.0

    baselines = np.zeros((n_joints, n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        baselines[j] = deform_path_to_start_and_goal(multi_bases[k, j], start_rot[j], goal_rot[j])
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device)

    base_f = torch.zeros(1, n_joints, 6, device=device)
    for j in range(n_joints):
        bj = torch.tensor(multi_bases[k, j], dtype=torch.float32, device=device)
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
        logits, mu, log_sigma = model(start_t, goal_t, action_t, base_f, bid_t)
        k_comp = int(logits.mean(dim=1).argmax(dim=-1)[0])
        residual_flat = mu[:, :, k_comp, :]
        residual = residual_flat.view(1, n_joints, T, 3)
        pb_generated = baselines_t.unsqueeze(0) + residual
    pb_np = pb_generated[0].permute(1, 0, 2).cpu().numpy()

    from scipy.signal import savgol_filter
    w = min(7, T - 2 if (T - 2) % 2 == 1 else T - 3)
    if w >= 5:
        for j in range(n_joints):
            for d in range(3):
                pb_np[:, j, d] = savgol_filter(pb_np[:, j, d], w, 3)

    pb_flat = pb_np.reshape(T, -1)
    if len(root_orient) != T:
        idx = np.linspace(0, len(root_orient) - 1, T, dtype=int)
        root_orient = root_orient[idx]
        trans = trans[idx]
    joints_3d = fk_smpl(pb_flat, root_orient, trans)
    body_joints = joints_3d[:, POSE_BODY_TO_SMPL, :]
    return body_joints


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-clusters", type=int, default=4)
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
    n_train = int(0.8 * len(data))
    train_data = data[perm[:n_train]]
    test_data = data[perm[n_train:]]

    # Multi-base clustering on training data
    print(f"\nClustering {len(train_data)} walks into {args.n_clusters} articulation styles...")
    train_pb = train_data[:, :, :63].reshape(len(train_data), args.n_frames, -1)
    multi_bases, cluster_ids, artic_feat = compute_multi_bases(train_pb, n_clusters=args.n_clusters)
    print(f"Base shapes: {multi_bases.shape}  (K={args.n_clusters}, J={n_joints}, T={args.n_frames}, 3)")

    print(f"\nBuilding model: {n_joints} joints, {args.n_clusters}-base conditioning, 4-comp MDN + cross-attention...")
    model = PerJointRotationMDN_v2(n_joints=n_joints, n_timesteps=args.n_frames, n_components=4,
                                    n_bases=args.n_clusters, feature_dim=32, hidden_dim=128).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining in rotation space (multi-base)...")
    t0 = time.time()
    best_loss, _ = train_model(model, train_pb, multi_bases, cluster_ids,
                                n_epochs=args.n_epochs, batch_size=args.batch_size,
                                lr=args.lr, device=device)
    print(f"Training complete: {time.time()-t0:.1f}s, best_loss={best_loss:.4f}")

    # Test across all clusters
    print("\nTesting (rotation -> FK -> 3D) per cluster...")
    test_pb = test_data[:, :, :63].reshape(len(test_data), args.n_frames, n_joints, 3)
    all_rmse = []
    for k in range(args.n_clusters):
        # Find test samples closest to this cluster
        test_artic = compute_articulation_features(test_pb.reshape(len(test_data), args.n_frames, -1))
        # Use the first test sample for quick eval
        pb_test = test_pb[0]
        ro_test = test_data[0, :, 63:66]
        tr_test = test_data[0, :, 66:69]
        start_rot = pb_test[0]
        goal_rot = pb_test[-1]

        generated_3d = generate_motion(model, multi_bases, k, start_rot, goal_rot,
                                        ro_test, tr_test, n_frames=args.n_frames, device=device)
        true_3d = fk_smpl(pb_test.reshape(args.n_frames, -1), ro_test, tr_test)
        true_body = true_3d[:, POSE_BODY_TO_SMPL, :]

        rmse = np.sqrt(np.mean((generated_3d - true_body) ** 2))
        goal_err = np.sqrt(np.mean((generated_3d[-1] - true_body[-1]) ** 2))
        all_rmse.append(rmse)
        print(f"  Base {k}: RMSE={rmse:.4f}m  goal_err={goal_err:.4f}m")

    print(f"\n  Mean RMSE across bases: {np.mean(all_rmse):.4f}m")
    print(f"  Bone lengths GUARANTEED by FK (Layer 2)")
    print("\nDone.")


if __name__ == "__main__":
    main()
