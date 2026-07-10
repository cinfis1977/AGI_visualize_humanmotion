#!/usr/bin/env python3
"""
Stage5G — Arm-Aware Vertex Rotation Baseball-Port with rot-phase repair.

Purpose
-------
This is the arm-motion repair of Stage5F:

  * every SMPL joint is treated as a vertex with its own rotation trajectory;
  * every vertex has its own 10-base rotation manifold, not one collapsed mean base;
  * each base trajectory is start/goal deformed in rotation space, like the baseball base path;
  * the model learns per-vertex residuals from the interpolated deformed base manifold;
  * local vertex phase/time is encoded from rotation motion and aligned with global time tokens;
  * cross-attention pulls local rotation-time information before the per-vertex MDNs predict residuals;
  * FK is still the final layer, so bone lengths remain guaranteed.

Usage
-----
  python scripts/stage5g_arm_phase_repair_baseball_port.py --n-files 200 --n-epochs 200

Drop-in note
------------
This script expects `smpl_fk.py` next to the script or importable from the project scripts folder.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None

from smpl_fk import fk_smpl_full

Array = np.ndarray

JOINT_NAMES_24 = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand",
]


# ============================================================
# Data loader — 24 joints: root(3) + body(63) + hands(6) + trans(3) = 75
# ============================================================

def load_amass_pose_body(
    data_dir: str,
    n_files: int = 200,
    n_frames: int = 96,
    min_translation: float = 1.0,
    seed: int = 42,
) -> Tuple[Array, int]:
    """Load AMASS walks and return 24-joint axis-angle rotations + translation.

    Output per frame: [root_orient(3), pose_body(63), hand_zeros(6), trans(3)] = 75-d.
    """
    base = Path(data_dir)
    all_files = sorted(base.rglob("Walk*.npz"))
    rng = np.random.default_rng(seed)

    sequences = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]
            tr = d["trans"]
            T_orig = int(pb.shape[0])
            if T_orig < n_frames:
                continue
            x_range = float(tr[:, 0].max() - tr[:, 0].min())
            if x_range < min_translation:
                continue
            idx = np.linspace(0, T_orig - 1, n_frames, dtype=int)
            pb_ds = pb[idx].astype(np.float32)
            ro_ds = d["root_orient"][idx].astype(np.float32)
            tr_ds = tr[idx].astype(np.float32)
            hand_z = np.zeros((n_frames, 6), dtype=np.float32)
            full_rot = np.concatenate([ro_ds, pb_ds, hand_z], axis=1)  # (T,72)
            sequences.append(np.concatenate([full_rot, tr_ds], axis=1))  # (T,75)
        except Exception:
            continue

    if not sequences:
        raise RuntimeError(f"No usable Walk*.npz files found under {base}")

    if n_files > 0 and len(sequences) > n_files:
        picks = rng.choice(len(sequences), n_files, replace=False)
        sequences = [sequences[int(i)] for i in picks]

    data = np.stack(sequences, axis=0).astype(np.float32)
    n_joints = 24
    print(f"Loaded {len(sequences)} straight walks (24 joints)")
    return data, n_joints


# ============================================================
# Baseball-style per-vertex base bank + deformation
# ============================================================

def compute_per_vertex_base_bank(data_rot: Array, n_base_walks: int = 10) -> Array:
    """Return a per-joint base bank chosen independently for every joint.

    Stage5F used the same first B walks for every vertex. That can leave the arm
    vertices with weak/static bases even when the legs move well. Stage5G chooses
    each vertex bank from the highest-amplitude/diverse rotation trajectories for
    that vertex. This is closer to the baseball rule: every vertex has its own
    base trajectory training set.

    Args:
        data_rot: (N,T,72) rotation data.
    Returns:
        base_bank: (J,B,T,3), where B<=n_base_walks.
    """
    N, T, D = data_rot.shape
    if D % 3 != 0:
        raise ValueError(f"Rotation dim must be divisible by 3; got {D}")
    J = D // 3
    B = min(n_base_walks, N)
    rot = data_rot.reshape(N, T, J, 3)
    bank = np.zeros((J, B, T, 3), dtype=np.float32)
    chosen = np.zeros((J, B), dtype=np.int64)
    amp_by_joint = np.std(rot, axis=1).mean(axis=-1)  # (N,J)

    ARM_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21}
    for j in range(J):
        amp = amp_by_joint[:, j]
        # For arms, prioritize motion amplitude even more strongly; for other
        # joints keep amplitude + diversity. This prevents the shoulder/collar
        # bank from becoming almost static.
        order = list(np.argsort(-amp))
        picks = []
        if j in ARM_JOINTS:
            picks = order[:B]
        else:
            # Greedy farthest selection among high-amplitude candidates.
            pool = order[:max(B * 4, B)]
            picks = [pool[0]]
            while len(picks) < B:
                best_i, best_score = None, -1.0
                for i in pool:
                    if i in picks:
                        continue
                    d = np.mean([np.sqrt(np.mean((rot[i, :, j] - rot[k, :, j]) ** 2)) for k in picks])
                    score = float(d + 0.25 * amp[i])
                    if score > best_score:
                        best_score, best_i = score, i
                picks.append(best_i if best_i is not None else pool[len(picks) % len(pool)])
        chosen[j, :len(picks)] = picks[:B]
        bank[j] = rot[picks[:B], :, j]

    amp = np.std(bank, axis=2).mean(axis=-1)  # (J,B)
    diversity = np.std(bank, axis=1).mean(axis=(1, 2))  # (J,)
    print(f"  Per-vertex AMPLITUDE-SELECTED base bank: J={J}, B={B}, T={T}")
    print(f"  Mean bank temporal amp={amp.mean():.5f}; arm amp={amp[list(ARM_JOINTS)].mean():.5f}")
    print(f"  Mean base-to-base diversity={diversity.mean():.5f}; arm diversity={diversity[list(ARM_JOINTS)].mean():.5f}")
    for j in sorted(ARM_JOINTS):
        if j < J:
            print(f"    arm bank {j:02d} {JOINT_NAMES_24[j]:>10s}: amp={amp[j].mean():.5f} div={diversity[j]:.5f} picks={chosen[j].tolist()}")
    return bank.astype(np.float32)


def deform_path_to_start_and_goal(path: Array, start: Array, goal: Array, gamma: float = 1.0) -> Array:
    """D_{s,g}[Q] — baseball-style endpoint deformation in rotation space."""
    Q = np.asarray(path, dtype=np.float32)
    s = np.asarray(start, dtype=np.float32).reshape(1, -1)
    g = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    alpha = np.linspace(0.0, 1.0, Q.shape[0], dtype=np.float32).reshape(-1, 1)
    return Q - Q[0:1] + s + gamma * alpha * (g - (Q[-1:] - Q[0:1] + s))


def build_deformed_base_candidates(base_bank: Array, starts: Array, goals: Array) -> Array:
    """Build deformed candidate base rotations.

    Args:
        base_bank: (J,B,T,3)
        starts/goals: (N,J,3)
    Returns:
        candidates: (N,J,B,T,3)
    """
    N, J = starts.shape[:2]
    Jb, B, T, _ = base_bank.shape
    if J != Jb:
        raise ValueError(f"starts has J={J}, base_bank has J={Jb}")
    out = np.empty((N, J, B, T, 3), dtype=np.float32)
    for n in range(N):
        for j in range(J):
            s_j = starts[n, j]
            g_j = goals[n, j]
            for b in range(B):
                out[n, j, b] = deform_path_to_start_and_goal(base_bank[j, b], s_j, g_j)
    return out


# ============================================================
# Local vertex phase/time ↔ global time tokenization
# ============================================================

def _safe_norm(x: Array, axis=-1, keepdims=False) -> Array:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims).astype(np.float32)


def compute_candidate_descriptors(candidates: Array) -> Array:
    """Candidate shape/phase descriptors from deformed base paths.

    Args:
        candidates: (N,J,B,T,3)
    Returns:
        desc: (N,J,B,14)
    """
    N, J, B, T, _ = candidates.shape
    desc = np.zeros((N, J, B, 14), dtype=np.float32)
    q = candidates
    d1 = q[..., 1:, :] - q[..., :-1, :]
    length = _safe_norm(d1, axis=-1).sum(axis=-1)
    d2 = q[..., 2:, :] - 2.0 * q[..., 1:-1, :] + q[..., :-2, :]
    curv = _safe_norm(d2, axis=-1).mean(axis=-1) / np.maximum(length, 1e-8)
    alpha = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(1, 1, 1, T, 1)
    chord = q[..., 0:1, :] + alpha * (q[..., -1:, :] - q[..., 0:1, :])
    dev = _safe_norm(q - chord, axis=-1).max(axis=-1)
    rot_angle = _safe_norm(q, axis=-1)

    ph_max_rot = np.argmax(rot_angle, axis=-1).astype(np.float32) / max(T - 1, 1)
    ph_min_rot = np.argmin(rot_angle, axis=-1).astype(np.float32) / max(T - 1, 1)
    ph_max_z = np.argmax(q[..., 2], axis=-1).astype(np.float32) / max(T - 1, 1)
    ph_min_z = np.argmin(q[..., 2], axis=-1).astype(np.float32) / max(T - 1, 1)

    desc[..., 0] = length
    desc[..., 1] = curv
    desc[..., 2] = dev
    desc[..., 3] = q[..., 2].max(axis=-1)
    desc[..., 4] = q[..., 2].std(axis=-1)
    desc[..., 5] = q[..., -1, 2] - q[..., 0, 2]
    desc[..., 6] = np.sin(2 * np.pi * ph_max_rot)
    desc[..., 7] = np.cos(2 * np.pi * ph_max_rot)
    desc[..., 8] = np.sin(2 * np.pi * ph_min_rot)
    desc[..., 9] = np.cos(2 * np.pi * ph_min_rot)
    desc[..., 10] = np.sin(2 * np.pi * ph_max_z)
    desc[..., 11] = np.cos(2 * np.pi * ph_max_z)
    desc[..., 12] = np.sin(2 * np.pi * ph_min_z)
    desc[..., 13] = np.cos(2 * np.pi * ph_min_z)
    return desc


def compute_local_global_phase_tokens(candidates: Array) -> Array:
    """Tokenize local vertex phase and global time for cross-attention.

    For every deformed base candidate, every global frame t becomes a token:
      rot_xyz, d_rot_xyz, local_phase_sin/cos, global_time_sin/cos,
      rot_angle, local_phase, base_id_sin/cos.

    Args:
        candidates: (N,J,B,T,3)
    Returns:
        tokens: (N,J,B*T,14)
    """
    N, J, B, T, _ = candidates.shape
    q = candidates.astype(np.float32)
    vel = np.zeros_like(q)
    vel[..., 1:, :] = q[..., 1:, :] - q[..., :-1, :]
    step = _safe_norm(vel, axis=-1)
    cum = np.cumsum(step, axis=-1)
    denom = np.maximum(cum[..., -1:], 1e-8)
    local_phase = cum / denom

    global_t = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(1, 1, 1, T)
    base_id = np.arange(B, dtype=np.float32).reshape(1, 1, B, 1) / max(B - 1, 1)
    rot_angle = _safe_norm(q, axis=-1)

    tok = np.zeros((N, J, B, T, 14), dtype=np.float32)
    tok[..., 0:3] = q
    tok[..., 3:6] = vel
    tok[..., 6] = np.sin(2 * np.pi * local_phase)
    tok[..., 7] = np.cos(2 * np.pi * local_phase)
    tok[..., 8] = np.sin(2 * np.pi * global_t)
    tok[..., 9] = np.cos(2 * np.pi * global_t)
    tok[..., 10] = rot_angle
    tok[..., 11] = local_phase
    tok[..., 12] = np.sin(2 * np.pi * base_id)
    tok[..., 13] = np.cos(2 * np.pi * base_id)
    return tok.reshape(N, J, B * T, 14)


# ============================================================
# Model
# ============================================================

class StandardScalerNP:
    def __init__(self, eps: float = 1e-9):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, X: Array) -> "StandardScalerNP":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = (X.std(axis=0) + self.eps).astype(np.float32)
        return self

    def transform(self, X: Array) -> Array:
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.std


class OneShotJointMDN(nn.Module):
    def __init__(self, feature_dim: int, n_timesteps: int = 96, n_components: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.target_dim = n_timesteps * 3
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.logits = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * self.target_dim)
        self.log_sigma = nn.Linear(hidden_dim, n_components)
        nn.init.zeros_(self.mu.bias)
        nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.log_sigma.bias)

    def forward(self, x):
        h = self.net(x)
        K = self.logits.out_features
        return self.logits(h), self.mu(h).view(-1, K, self.target_dim), self.log_sigma(h).clamp(-4.0, 3.0)


class VertexRotPhaseArmRepairMDN(nn.Module):
    """Per-vertex rotation MDNs with base-bank interpolation and rot-phase cross-attention."""

    def __init__(
        self,
        n_joints: int = 24,
        n_bases: int = 10,
        n_timesteps: int = 96,
        n_components: int = 4,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        token_dim: int = 14,
        desc_dim: int = 14,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_bases = n_bases
        self.n_timesteps = n_timesteps
        self.n_components = n_components
        self.feature_dim = feature_dim

        # Geometry query: start_rot + goal_rot + action = 9 dims.
        self.geo_encoder = nn.Sequential(
            nn.Linear(9, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
        )
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
        )

        # Local phase attention: per vertex, query is global joint state;
        # keys/values are local-vertex phase tokens over B base paths × T global frames.
        self.local_phase_attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)

        # Cross-joint attention: after local time alignment, vertices can pull rotation context from each other.
        self.cross_joint_attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)

        # Learned base interpolation weights per vertex.
        self.base_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_bases),
        )

        mdn_in = feature_dim * 3 + n_bases  # geo + local_phase + cross_joint + base weights
        self.mdns = nn.ModuleList([
            OneShotJointMDN(mdn_in, n_timesteps=n_timesteps, n_components=n_components, hidden_dim=hidden_dim)
            for _ in range(n_joints)
        ])

    def forward(self, start_rot, goal_rot, action, base_desc, phase_tokens):
        """Forward.

        Args:
            start_rot/goal_rot: (B,J,3), standardized outside.
            action: (B,3)
            base_desc: (B,J,NBASE,14)
            phase_tokens: (B,J,NBASE*T,14)
        Returns:
            base_logits: (B,J,NBASE)
            mdn_logits: (B,J,K)
            mu: (B,J,K,T*3) raw residual in axis-angle units
            log_sigma: (B,J,K)
        """
        Bsz, J = start_rot.shape[:2]
        act = action.unsqueeze(1).expand(Bsz, J, 3)
        geo_in = torch.cat([start_rot, goal_rot, act], dim=-1)
        geo = self.geo_encoder(geo_in)  # (B,J,F)

        # Summarize base descriptors, but keep the full bank for learned base weights.
        desc_emb = self.desc_encoder(base_desc)  # (B,J,NB,F)
        desc_pool = desc_emb.mean(dim=2)         # (B,J,F)

        # Local phase/time attention per vertex.
        # Flatten vertices so each joint attends only to its own local phase tokens.
        q = (geo + desc_pool).reshape(Bsz * J, 1, self.feature_dim)
        tok = self.token_encoder(phase_tokens).reshape(Bsz * J, -1, self.feature_dim)
        local_ctx, _ = self.local_phase_attn(q, tok, tok, need_weights=False)
        local_ctx = local_ctx.reshape(Bsz, J, self.feature_dim)

        # Cross-joint attention after local→global alignment.
        joint_tokens = geo + local_ctx
        cross_ctx, _ = self.cross_joint_attn(joint_tokens, joint_tokens, joint_tokens, need_weights=False)

        gate_input = torch.cat([geo, local_ctx, cross_ctx], dim=-1)
        base_logits = self.base_gate(gate_input)  # (B,J,NB)
        base_weights = torch.softmax(base_logits, dim=-1)

        mdn_input = torch.cat([geo, local_ctx, cross_ctx, base_weights], dim=-1)
        all_logits, all_mu, all_sigma = [], [], []
        for j in range(J):
            l, m, s = self.mdns[j](mdn_input[:, j, :])
            all_logits.append(l.unsqueeze(1))
            all_mu.append(m.unsqueeze(1))
            all_sigma.append(s.unsqueeze(1))
        return base_logits, torch.cat(all_logits, 1), torch.cat(all_mu, 1), torch.cat(all_sigma, 1)


# ============================================================
# Training / generation
# ============================================================

def _make_action(n: int, device: str):
    a = torch.zeros(n, 3, device=device)
    a[:, 0] = 1.0
    return a


def _batch_tensors(idx, arrays, device):
    return [torch.tensor(a[idx], dtype=torch.float32, device=device) for a in arrays]


def get_joint_loss_weights(n_joints: int, device: str = "cpu"):
    """Arm-aware weights.

    Legs already moved in Stage5F; shoulders/collars/elbows/wrists were being
    under-optimized because they are only a small subset of 24 joints. Give the
    arm chain more loss mass and keep hands low because AMASS hand rotations are
    synthetic zeros here.
    """
    w = np.ones(n_joints, dtype=np.float32)
    for j in [13, 14]:          # collars / clavicles: shoulder root carrier
        if j < n_joints: w[j] = 3.0
    for j in [16, 17]:          # shoulders: main missing driver
        if j < n_joints: w[j] = 5.0
    for j in [18, 19]:          # elbows
        if j < n_joints: w[j] = 4.0
    for j in [20, 21]:          # wrists
        if j < n_joints: w[j] = 3.0
    for j in [22, 23]:          # hand rotations are zero-filled in this loader
        if j < n_joints: w[j] = 0.25
    w = w / max(w.mean(), 1e-8)
    return torch.tensor(w, dtype=torch.float32, device=device)


def residual_endpoint_taper(T: int, device: str = "cpu"):
    """Baseball-style endpoint lock for residuals: residual is zero at t=0,T-1."""
    a = torch.linspace(0.0, 1.0, T, device=device)
    return (4.0 * a * (1.0 - a)).clamp_min(0.0).view(1, 1, 1, T, 1)


def train_model(
    model: VertexRotPhaseArmRepairMDN,
    data_rot: Array,
    base_bank: Array,
    n_epochs: int = 200,
    batch_size: int = 12,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train in rotation space with baseball-style base-bank interpolation + per-joint MCL."""
    n_samples, T, dim = data_rot.shape
    J = model.n_joints
    X_np = data_rot.reshape(n_samples, T, J, 3).astype(np.float32)
    starts_np = X_np[:, 0]
    goals_np = X_np[:, -1]

    print("  Building deformed 10-base candidates per vertex...")
    candidates_np = build_deformed_base_candidates(base_bank, starts_np, goals_np)
    desc_np = compute_candidate_descriptors(candidates_np)
    tokens_np = compute_local_global_phase_tokens(candidates_np)

    # Scale only start/goal rotation endpoints. Phase/base tokens stay in physical units.
    x_scaler = StandardScalerNP().fit(np.concatenate([starts_np, goals_np], axis=-1).reshape(-1, 6))
    sg_scaled = x_scaler.transform(np.concatenate([starts_np, goals_np], axis=-1).reshape(-1, 6)).reshape(n_samples, J, 6)
    starts_scaled_np = sg_scaled[..., :3]
    goals_scaled_np = sg_scaled[..., 3:]

    # Deterministic validation split.
    n_val = max(1, int(0.15 * n_samples))
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    print(f"  Train={len(tr_idx)} val={len(val_idx)}; candidates={candidates_np.shape}; tokens={tokens_np.shape}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    joint_w = get_joint_loss_weights(J, device=device).view(1, J)
    taper = residual_endpoint_taper(T, device=device)
    vel_weight = 0.35
    best_state = None
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        total = 0.0
        count = 0
        for st in range(0, len(tr_idx), batch_size):
            b_idx = tr_idx[st:st + batch_size]
            bs = len(b_idx)
            start_b, goal_b, desc_b, tok_b, cand_b, true_b = _batch_tensors(
                b_idx,
                [starts_scaled_np, goals_scaled_np, desc_np, tokens_np, candidates_np, X_np.transpose(0, 2, 1, 3)],
                device,
            )
            act_b = _make_action(bs, device)

            base_logits, mdn_logits, mu, _ = model(start_b, goal_b, act_b, desc_b, tok_b)
            base_w = torch.softmax(base_logits, dim=-1)  # (B,J,NB)
            baseline = torch.einsum("bjn,bjntd->bjtd", base_w, cand_b)
            residual = mu.view(bs, J, model.n_components, T, 3) * taper
            pred = baseline.unsqueeze(2) + residual

            pos_err = ((pred - true_b.unsqueeze(2)) ** 2).mean(dim=(3, 4))  # (B,J,K)
            pred_vel = pred[..., 1:, :] - pred[..., :-1, :]
            true_vel = true_b[:, :, 1:, :] - true_b[:, :, :-1, :]
            vel_err = ((pred_vel - true_vel.unsqueeze(2)) ** 2).mean(dim=(3, 4))
            comp_err = pos_err + vel_weight * vel_err
            winner = comp_err.argmin(dim=-1)
            eps = 0.05
            w = torch.full_like(comp_err, eps / max(model.n_components - 1, 1))
            w.scatter_(-1, winner.unsqueeze(-1), 1.0 - eps)
            joint_loss = (w * comp_err).sum(dim=-1)  # (B,J)
            recon = (joint_loss * joint_w).mean()
            gate_terms = []
            for j in range(J):
                gate_terms.append(F.cross_entropy(mdn_logits[:, j, :], winner[:, j]) * joint_w[0, j])
            gate = torch.stack(gate_terms).mean()
            # Mild entropy penalty avoids a completely flat base interpolation but keeps diversity.
            base_entropy = -(base_w * (base_w + 1e-8).log()).sum(dim=-1).mean()
            loss = 100.0 * recon + 0.1 * gate + 0.001 * base_entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * bs
            count += bs

        # validation
        model.eval()
        with torch.no_grad():
            vals = []
            for st in range(0, len(val_idx), batch_size):
                b_idx = val_idx[st:st + batch_size]
                bs = len(b_idx)
                start_b, goal_b, desc_b, tok_b, cand_b, true_b = _batch_tensors(
                    b_idx,
                    [starts_scaled_np, goals_scaled_np, desc_np, tokens_np, candidates_np, X_np.transpose(0, 2, 1, 3)],
                    device,
                )
                act_b = _make_action(bs, device)
                base_logits, mdn_logits, mu, _ = model(start_b, goal_b, act_b, desc_b, tok_b)
                base_w = torch.softmax(base_logits, dim=-1)
                baseline = torch.einsum("bjn,bjntd->bjtd", base_w, cand_b)
                residual = mu.view(bs, J, model.n_components, T, 3) * taper
                pred = baseline.unsqueeze(2) + residual
                pos_err = ((pred - true_b.unsqueeze(2)) ** 2).mean(dim=(3, 4))
                pred_vel = pred[..., 1:, :] - pred[..., :-1, :]
                true_vel = true_b[:, :, 1:, :] - true_b[:, :, :-1, :]
                vel_err = ((pred_vel - true_vel.unsqueeze(2)) ** 2).mean(dim=(3, 4))
                comp_err = pos_err + vel_weight * vel_err
                vals.append(float((comp_err.min(dim=-1).values * joint_w).mean().item()))
            val = float(np.mean(vals)) if vals else float("inf")

        if val < best_val - 1e-7:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == n_epochs:
            print(f"  epoch {epoch:4d}: train={total/max(count,1):.5f}  val={val:.6f}  best={best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, x_scaler


def generate_motion(
    model: VertexRotPhaseArmRepairMDN,
    base_bank: Array,
    start_rot: Array,
    goal_rot: Array,
    trans: Array,
    x_scaler: StandardScalerNP | None = None,
    n_frames: int = 96,
    device: str = "cpu",
):
    """Generate 24-joint axis-angle rotations, then FK to 3D positions."""
    model.eval()
    J = model.n_joints
    start_np = np.asarray(start_rot, dtype=np.float32).reshape(J, 3)
    goal_np = np.asarray(goal_rot, dtype=np.float32).reshape(J, 3)

    candidates = build_deformed_base_candidates(base_bank, start_np[None], goal_np[None])
    desc = compute_candidate_descriptors(candidates)
    tokens = compute_local_global_phase_tokens(candidates)

    sg = np.concatenate([start_np, goal_np], axis=-1)
    if x_scaler is not None:
        sg = x_scaler.transform(sg)
    start_s = torch.tensor(sg[:, :3][None], dtype=torch.float32, device=device)
    goal_s = torch.tensor(sg[:, 3:][None], dtype=torch.float32, device=device)
    desc_t = torch.tensor(desc, dtype=torch.float32, device=device)
    tok_t = torch.tensor(tokens, dtype=torch.float32, device=device)
    cand_t = torch.tensor(candidates, dtype=torch.float32, device=device)
    act = _make_action(1, device)

    with torch.no_grad():
        base_logits, mdn_logits, mu, _ = model(start_s, goal_s, act, desc_t, tok_t)
        base_w = torch.softmax(base_logits, dim=-1)
        baseline = torch.einsum("bjn,bjntd->bjtd", base_w, cand_t)  # (1,J,T,3)
        # Use expectation rather than hard argmax for arm repair; hard argmax was
        # observed to select static shoulder components too often.
        comp_w = torch.softmax(mdn_logits, dim=-1)  # (1,J,K)
        residual_all = mu.view(1, J, model.n_components, n_frames, 3)
        taper = residual_endpoint_taper(n_frames, device=device)
        residual = (comp_w.view(1, J, model.n_components, 1, 1) * residual_all * taper).sum(dim=2)
        rot_pred = (baseline + residual)[0].permute(1, 0, 2).cpu().numpy()  # (T,J,3)

    full_pose = rot_pred.reshape(n_frames, 72)
    if len(trans) != n_frames:
        idx = np.linspace(0, len(trans) - 1, n_frames, dtype=int)
        trans = trans[idx]
    joints_3d = fk_smpl_full(full_pose, trans)
    return joints_3d, rot_pred


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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if torch is None:
        print("PyTorch not available.")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print("Loading AMASS straight walks...")
    data, n_joints = load_amass_pose_body(args.data_dir, args.n_files, args.n_frames, seed=args.seed)
    print(f"Data: {data.shape}")

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(data))

    base_data = data[perm[:args.n_base_walks]]
    rot_base = base_data[:, :, :72]

    print(f"\nComputing per-vertex 10-base rotation banks from {len(base_data)} walks...")
    base_bank = compute_per_vertex_base_bank(rot_base, n_base_walks=args.n_base_walks)

    remaining = data[perm[args.n_base_walks:]]
    n_train = int(0.8 * len(remaining))
    train_data = remaining[:n_train]
    test_data = remaining[n_train:]
    print(f"Base: {len(base_data)}, Train: {len(train_data)}, Test: {len(test_data)}")

    n_bases = base_bank.shape[1]
    print(f"\nBuilding Stage5G arm-repair model: {n_joints} vertices, {n_bases} base paths/vertex, 4-comp MDN...")
    model = VertexRotPhaseArmRepairMDN(
        n_joints=n_joints,
        n_bases=n_bases,
        n_timesteps=args.n_frames,
        n_components=4,
        feature_dim=64,
        hidden_dim=128,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining arm-aware per-vertex rotation baseball port...")
    t0 = time.time()
    train_rot = train_data[:, :, :72]
    best_val, x_scaler = train_model(
        model,
        train_rot,
        base_bank,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    print(f"Training complete: {time.time() - t0:.1f}s, best_val={best_val:.6f}")

    if len(test_data) == 0:
        print("No test data after split.")
        return

    print("\nTesting (rotation -> FK -> 3D, ALL 24 joints)...")
    test = test_data[0]
    rot_test = test[:args.n_frames, :72].reshape(args.n_frames, n_joints, 3)
    tr_test = test[:args.n_frames, 72:75]
    start_rot = rot_test[0]
    goal_rot = rot_test[-1]

    generated_3d, gen_rot = generate_motion(
        model,
        base_bank,
        start_rot,
        goal_rot,
        tr_test,
        x_scaler=x_scaler,
        n_frames=args.n_frames,
        device=device,
    )
    true_3d = fk_smpl_full(rot_test.reshape(args.n_frames, 72), tr_test)

    rmse = float(np.sqrt(np.mean((generated_3d - true_3d) ** 2)))
    goal_err = float(np.sqrt(np.mean((generated_3d[-1] - true_3d[-1]) ** 2)))
    per_joint_rmse = np.sqrt(np.mean((generated_3d - true_3d) ** 2, axis=(0, 2)))

    print(f"  Position RMSE (24 joints): {rmse:.4f}m")
    print(f"  Goal error: {goal_err:.4f}m")
    print("  Bone lengths GUARANTEED by FK (Layer 2)")

    gen_amp = float(np.std(generated_3d, axis=0).mean())
    true_amp = float(np.std(true_3d, axis=0).mean())
    ratio = gen_amp / max(true_amp, 1e-8)
    print(f"\n  Articulation amplitude: gen={gen_amp:.4f} true={true_amp:.4f} ratio={ratio:.2f}")
    print("  [OK] Articulation recovery GOOD (>70% of true amplitude)" if ratio > 0.7 else "  [WARN] Articulation still LOW")

    # Arm-specific diagnostics: this directly checks the failure the user saw.
    arm_joints = [13,14,16,17,18,19,20,21]
    arm_joints = [j for j in arm_joints if j < n_joints]
    leg_joints = [1,2,4,5,7,8,10,11]
    leg_joints = [j for j in leg_joints if j < n_joints]
    arm_amp_gen = float(np.std(generated_3d[:, arm_joints, :], axis=0).mean()) if arm_joints else 0.0
    arm_amp_true = float(np.std(true_3d[:, arm_joints, :], axis=0).mean()) if arm_joints else 1.0
    leg_amp_gen = float(np.std(generated_3d[:, leg_joints, :], axis=0).mean()) if leg_joints else 0.0
    leg_amp_true = float(np.std(true_3d[:, leg_joints, :], axis=0).mean()) if leg_joints else 1.0
    print(f"\n  ARM amplitude: gen={arm_amp_gen:.4f} true={arm_amp_true:.4f} ratio={arm_amp_gen/max(arm_amp_true,1e-8):.2f}")
    print(f"  LEG amplitude: gen={leg_amp_gen:.4f} true={leg_amp_true:.4f} ratio={leg_amp_gen/max(leg_amp_true,1e-8):.2f}")
    for j in [13,14,16,17,18,19,20,21]:
        if j < n_joints:
            rot_amp_g = float(np.std(gen_rot[:, j, :], axis=0).mean())
            rot_amp_t = float(np.std(rot_test[:, j, :], axis=0).mean())
            print(f"    {j:02d} {JOINT_NAMES_24[j]:>10s}: rot_amp gen={rot_amp_g:.5f} true={rot_amp_t:.5f} ratio={rot_amp_g/max(rot_amp_t,1e-8):.2f}")

    top3 = np.argsort(per_joint_rmse)[-3:][::-1]
    top3_names = [JOINT_NAMES_24[int(i)] for i in top3]
    print(f"\n  Highest-error joints: {top3_names} ({per_joint_rmse[top3]}m)")
    print("\nDone.")


if __name__ == "__main__":
    main()
