#!/usr/bin/env python3
"""
Stage5G — Model A: PHASE-CONDITIONED residual + differentiable FK (position-space loss).

Two-model plan, part A. Each joint's residual is a function of a SHARED gait phase
phi(t): residual_j(t) = decoder([context_j, phaseenc(phi(t))]). Because every joint
reads the SAME phi(t), the limbs are phase-locked (arms swing in sync with legs) — the
thing the per-joint one-shot model could not do. phi resolves the gait-phase ambiguity
that made the endpoint-only MDN multi-modal, so no MDN/MCL is needed here; the stochastic
"imagination" will live in Model B (the phi generator), built next.

Layer 1: phase-conditioned residual over a deformed MEAN base (rotations).
Layer 2: differentiable smpl_fk_torch -> 3D positions (bone lengths guaranteed).
Loss:    POSITION space (+ endpoint term).

Phase 1 test: reconstruction with the TRUE phi(t) — does arm-swing timing lift end-to-end?

Does NOT modify stage5e files; reuses their data/base/deform helpers.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import stage5e_per_joint_10base_mdn as s5e
from stage5f_phase_probe import gait_phase, swing_dir, swing, phase_profile, eval_at_phase
from smpl_fk_torch import fk_smpl_torch

JOINT_NAMES_24 = ["pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2",
                  "L_ankle","R_ankle","spine3","L_foot","R_foot","neck",
                  "L_collar","R_collar","head","L_shoulder","R_shoulder",
                  "L_elbow","R_elbow","L_wrist","R_wrist","L_hand","R_hand"]


# ============================================================
# Model A
# ============================================================

class PhaseConditionedResidual(nn.Module):
    def __init__(self, n_joints=24, feature_dim=32, hidden=128, n_harmonics=4):
        super().__init__()
        self.n_joints = n_joints
        self.H = n_harmonics
        self.joint_encoder = nn.Sequential(
            nn.Linear(15, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        dec_in = feature_dim * 2 + 2 * n_harmonics
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.normal_(self.decoder[-1].weight, 0.0, 1e-3)

    def phase_encode(self, phi):                        # phi (B,T) -> (B,T,2H)
        parts = []
        for m in range(1, self.H + 1):
            parts += [torch.sin(m * phi), torch.cos(m * phi)]
        return torch.stack(parts, dim=-1)

    def forward(self, start, goal, action, base_feats, phi):
        # start,goal (B,J,3); base_feats (B,J,6); action (B,3); phi (B,T)
        B, J = start.shape[:2]
        T = phi.shape[1]
        act = action.unsqueeze(1).expand(B, J, 3)
        feat = torch.cat([start, goal, base_feats, act], dim=-1)     # (B,J,15)
        enc = self.joint_encoder(feat)
        ctx, _ = self.attn(enc, enc, enc)
        context = torch.cat([enc, ctx], dim=-1)                      # (B,J,2F)
        pe = self.phase_encode(phi)                                  # (B,T,2H)
        ctx_e = context.unsqueeze(2).expand(B, J, T, context.shape[-1])
        pe_e = pe.unsqueeze(1).expand(B, J, T, pe.shape[-1])
        res = self.decoder(torch.cat([ctx_e, pe_e], dim=-1))         # (B,J,T,3)
        return res


# ============================================================
# Helpers
# ============================================================

def compute_base_features(bases_rot, T):
    n_joints = len(bases_rot)
    bf = np.zeros((n_joints, 6), np.float32)
    for j in range(n_joints):
        bj = bases_rot[j]
        d1 = bj[1:] - bj[:-1]; length = np.linalg.norm(d1, axis=-1).sum()
        d2 = bj[2:] - 2*bj[1:-1] + bj[:-2]
        curv = np.linalg.norm(d2, axis=-1).mean() / max(length, 1e-8) if len(d2) > 0 else 0.0
        chord = bj[0:1] + np.linspace(0, 1, T).reshape(-1, 1) * (bj[-1:] - bj[0:1])
        dev = np.linalg.norm(bj - chord, axis=-1).max()
        bf[j, 0] = length; bf[j, 1] = curv; bf[j, 2] = dev
        bf[j, 3] = bj[:, 2].max(); bf[j, 4] = bj[:, 2].std(); bf[j, 5] = bj[-1, 2] - bj[0, 2]
    return bf


def fk_positions(pose_rot, trans):
    """pose_rot (B,J=24,T,3), trans (B,T,3) -> positions (B,T,24,3)."""
    B, J, T, _ = pose_rot.shape
    pr = pose_rot.permute(0, 2, 1, 3)               # (B,T,24,3)
    root = pr[..., 0, :]                            # (B,T,3)
    body = pr[..., 1:22, :].reshape(B, T, 63)       # (B,T,63)  body joints 1..21
    return fk_smpl_torch(body, root, trans)         # (B,T,24,3)


def build_phase_base(rot_base, P=64):
    """Phase-INDEXED base: average each joint's rotation as a function of gait phase.
    rot_base (n,T,24,3) -> (phase_base (P,24,3), grid (P,)). Averaging in PHASE (not
    absolute time) keeps the swing that time-averaging cancelled — this is the base
    restored to a properly-aligned temporal template.
    """
    profs, grid = [], None
    for i in range(len(rot_base)):
        phi = gait_phase(rot_base[i])
        prof, grid = phase_profile(rot_base[i], phi)
        profs.append(prof)
    return np.mean(profs, axis=0).astype(np.float32), grid


def deformed_baselines_phase(phase_base, grid, phi, starts, goals, gamma=1.0):
    """Per-sample base = phase_base evaluated at the sample's own phi(t), then
    endpoint-deformed. starts,goals (N,J,3), phi (N,T) -> baselines (N,J,T,3)."""
    N, J = starts.shape[:2]
    T = phi.shape[1]
    out = np.zeros((N, J, T, 3), np.float32)
    for s in range(N):
        traj = eval_at_phase(phase_base, grid, phi[s])              # (T,24,3)
        for j in range(J):
            out[s, j] = s5e.deform_path_to_start_and_goal(traj[:, j, :], starts[s, j], goals[s, j], gamma=gamma)
    return out


# ============================================================
# Train
# ============================================================

def train(model, rot, trans, phase_base, grid, x_scaler, base_feats_std,
          n_epochs=200, batch_size=32, lr=1e-3, gamma=1.0, device="cpu"):
    N, T, J, _ = rot.shape
    starts = rot[:, 0]; goals = rot[:, -1]                          # (N,J,3)

    # per-sample gait phase (anchored, consistent) from TRUE rotations
    phi = np.stack([gait_phase(rot[s]) for s in range(N)]).astype(np.float32)  # (N,T)
    phi_t = torch.tensor(phi, device=device)

    base = deformed_baselines_phase(phase_base, grid, phi, starts, goals, gamma)  # (N,J,T,3)
    base_t = torch.tensor(base, device=device)
    rot_t = torch.tensor(rot, device=device)                       # (N,T,J,3)
    trans_t = torch.tensor(trans, device=device)                   # (N,T,3)

    # residual target (rotation space) + per-joint/dim scale so EVERY joint is
    # weighted equally. Position-MSE structurally under-weights the arms (they barely
    # move in metres) and lets the arm swing collapse; equal-weight rotation loss fixes
    # that. Bone length is still guaranteed by FK at generation.
    res_target = rot_t.permute(0, 2, 1, 3) - base_t                 # (N,J,T,3)
    res_scale = res_target.std(dim=(0, 2), keepdim=True).clamp_min(1e-3)  # (1,J,1,3)

    # scaled static inputs
    sg = np.concatenate([starts, goals], axis=-1)                   # (N,J,6)
    sg_std = x_scaler.transform(sg.reshape(-1, 6)).reshape(N, J, 6).astype(np.float32)
    start_s = torch.tensor(sg_std[..., :3], device=device)
    goal_s = torch.tensor(sg_std[..., 3:], device=device)
    bf_t = torch.tensor(np.tile(base_feats_std[None], (N, 1, 1)), device=device)  # (N,J,6)
    action = torch.zeros(N, 3, device=device); action[:, 0] = 1.0

    n_val = max(1, int(0.15 * N))
    idx = torch.randperm(N)
    val, tr = idx[:n_val], idx[n_val:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf"); best_state = None

    def loss_for(b):
        res = model(start_s[b], goal_s[b], action[b], bf_t[b], phi_t[b])  # (B,J,T,3)
        return (((res - res_target[b]) / res_scale) ** 2).mean()    # equal-weight rotation residual

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for st in range(0, len(tr), batch_size):
            b = perm[st:st + batch_size]
            loss = loss_for(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = loss_for(val).item()
        if vloss < best - 1e-6:
            best = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}: train={tot/len(tr):.5f}  val={vloss:.5f}  best={best:.5f}")
    if best_state:
        model.load_state_dict(best_state)
    return best


# ============================================================
# Generate (reconstruction: caller supplies phi)
# ============================================================

def generate(model, phase_base, grid, start_rot, goal_rot, trans, phi, x_scaler, base_feats_std,
             gamma=1.0, device="cpu"):
    model.eval()
    J = start_rot.shape[0]; T = len(phi)
    base = deformed_baselines_phase(phase_base, grid, phi[None], start_rot[None], goal_rot[None], gamma)[0]  # (J,T,3)
    sg = np.concatenate([start_rot, goal_rot], axis=-1)
    sg_std = x_scaler.transform(sg.reshape(-1, 6)).reshape(1, J, 6).astype(np.float32)
    with torch.no_grad():
        res = model(torch.tensor(sg_std[..., :3], device=device),
                    torch.tensor(sg_std[..., 3:], device=device),
                    torch.tensor([[1.0, 0.0, 0.0]], device=device),
                    torch.tensor(base_feats_std[None], device=device),
                    torch.tensor(phi[None].astype(np.float32), device=device))[0]  # (J,T,3)
    pose = torch.tensor(base, device=device) + res                 # (J,T,3)
    pose_rot = pose.permute(1, 0, 2).cpu().numpy()                 # (T,J,3)
    return pose_rot


# ============================================================
# Main — train + reconstruction test on arm timing
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-base-walks", type=int, default=20)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, n_joints = s5e.load_amass_pose_body(
        "C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz",
        n_files=args.n_files, n_frames=args.n_frames, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    base_idx = rng.choice(len(data), args.n_base_walks, replace=False)
    rot_base = data[base_idx][:, :, :72].reshape(args.n_base_walks, args.n_frames, -1)
    bases_rot = s5e.compute_per_joint_mean_bases(rot_base, n_base_walks=args.n_base_walks)  # time-mean, only for base descriptors
    phase_base, grid = build_phase_base(rot_base.reshape(args.n_base_walks, args.n_frames, 24, 3))
    print(f"  Phase-indexed base built: {phase_base.shape}")

    mask = np.ones(len(data), bool); mask[base_idx] = False
    rest = data[mask]
    n_train = int(0.85 * len(rest))
    train_data, test_data = rest[:n_train], rest[n_train:]
    rot_tr = train_data[:, :, :72].reshape(len(train_data), args.n_frames, 24, 3)
    tr_tr = train_data[:, :, 72:75]

    base_feats = compute_base_features(bases_rot, args.n_frames)
    # scalers
    sg_all = np.concatenate([rot_tr[:, 0], rot_tr[:, -1]], -1).reshape(-1, 6)
    x_scaler = s5e.StandardScalerNP().fit(sg_all)
    bf_scaler = s5e.StandardScalerNP().fit(base_feats)
    base_feats_std = bf_scaler.transform(base_feats)

    model = PhaseConditionedResidual(n_joints=24, feature_dim=32, hidden=128, n_harmonics=4).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("Training (equal-weight rotation loss, PHASE-INDEXED base)...")
    t0 = time.time()
    train(model, rot_tr, tr_tr, phase_base, grid, x_scaler, base_feats_std,
          n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
          gamma=args.gamma, device=device)
    print(f"  done {time.time()-t0:.0f}s")

    # reconstruction test: feed the test walk's OWN phi
    tf = next(Path('C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz')
              .rglob('walking_medium01_stageii.npz'))
    dd = np.load(tf, allow_pickle=True)
    idx = np.linspace(0, dd['pose_body'].shape[0] - 1, args.n_frames, dtype=int)
    hz = np.zeros((args.n_frames, 6), np.float32)
    true_rot = np.concatenate([dd['root_orient'][idx], dd['pose_body'][idx], hz], axis=1).reshape(args.n_frames, 24, 3)
    tr_test = dd['trans'][idx]
    phi_test = gait_phase(true_rot)

    gen_rot = generate(model, phase_base, grid, true_rot[0], true_rot[-1], tr_test, phi_test,
                       x_scaler, base_feats_std, gamma=args.gamma, device=device)

    # position RMSE
    with torch.no_grad():
        gp = fk_smpl_torch(torch.tensor(gen_rot[None, :, 1:22, :].reshape(1, args.n_frames, 63)),
                           torch.tensor(gen_rot[None, :, 0, :]),
                           torch.tensor(tr_test[None].astype(np.float32)))[0].numpy()
        tp = fk_smpl_torch(torch.tensor(true_rot[None, :, 1:22, :].reshape(1, args.n_frames, 63)),
                           torch.tensor(true_rot[None, :, 0, :]),
                           torch.tensor(tr_test[None].astype(np.float32)))[0].numpy()
    rmse = np.sqrt(((gp - tp) ** 2).mean())
    print(f"\n  Reconstruction position RMSE: {rmse:.4f}m")

    print("\n=== SHOULDER swing timing (Model A, phase-conditioned) vs TRUE ===")
    print("  (compare against stage5e baseline: L~0.26, R~0.13)")
    for j, nm in [(16, "L_shoulder"), (17, "R_shoulder")]:
        v = swing_dir(true_rot, j)
        tS = swing(true_rot, j, v); gS = swing(gen_rot, j, v)
        c = np.corrcoef(gS, tS)[0, 1]
        print(f"  {nm}: swing_corr={c:+.3f}  amp gen={np.rad2deg(gS.std()):4.1f}deg true={np.rad2deg(tS.std()):4.1f}deg")


if __name__ == "__main__":
    main()
