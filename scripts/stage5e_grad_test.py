#!/usr/bin/env python3
"""
Gradient flow diagnostic: Does FK position loss produce
meaningful gradients for foot/ankle rotations?

Tests hypothesis:
  (A) Vanishing gradients → foot joints get near-zero gradient
  (B) Loss in wrong space → gradients exist but don't fix position

Works on ONE batch, no training — just forward + backward + measure.
"""
import sys, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch, torch.nn as nn
from smpl_fk_torch import fk_smpl_torch

from stage5e_fixed_v1 import (load_data, compute_mean_base, baseball_deform,
                               BaseballVertexMDN)

JOINT_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1",
    "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3",
    "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand"
]

def main():
    data_dir = "C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz"
    n_files, n_frames, n_base = 200, 96, 10
    seed, J = 42, 24
    device = "cpu"

    print("[GradTest] Loading data...")
    data, _ = load_data(data_dir, n_files, n_frames, seed=seed)
    rng = np.random.default_rng(seed+1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:n_base]]
    train_data = data[perm[n_base:]][:int(0.8*(len(data)-n_base))]

    bases = compute_mean_base(base_data[:,:,:72], n_base)
    rot_data = train_data[:,:,:72].reshape(-1, n_frames, J, 3)
    trans_data = train_data[:,:,72:75]

    # Pre-compute baselines
    print("[GradTest] Computing baselines...")
    N = len(train_data)
    bl = np.zeros((N, n_frames, J, 3), dtype=np.float32)
    for n in range(N):
        for j in range(J):
            bl[n,:,j] = baseball_deform(bases[j], rot_data[n,0,j], rot_data[n,-1,j])

    # Model
    model = BaseballVertexMDN(n_joints=J, n_timesteps=n_frames).to(device)
    model.eval()

    # Scaler setup (same as train_model)
    sg = np.concatenate([rot_data[:,0], rot_data[:,-1]], -1).reshape(N, J, 6)
    sg_mean = sg.reshape(-1,6).mean(0).astype(np.float32)
    sg_std = sg.reshape(-1,6).std(0).astype(np.float32) + 1e-8
    class ScalerObj:
        def __init__(self, m, s): self.mean=m; self.std=s
        def transform(self, X): return (X-self.mean)/self.std
        def inverse(self, X): return X*self.std+self.mean
    xs = ScalerObj(sg_mean, sg_std)

    sg_s = xs.transform(sg.reshape(-1,6)).reshape(N, J, 6)
    Ss = torch.tensor(sg_s[...,:3], dtype=torch.float32, device=device)[:1]
    Gs = torch.tensor(sg_s[...,3:], dtype=torch.float32, device=device)[:1]

    base_feat = np.zeros((N, J, 6), dtype=np.float32)
    for j in range(J):
        bj = bases[j]
        base_feat[:, j, 0] = np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum()
        base_feat[:, j, 1] = np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if n_frames>2 else 0
        base_feat[:, j, 3] = bj[:,2].max()
        base_feat[:, j, 4] = bj[:,2].std()
        base_feat[:, j, 5] = bj[-1,2] - bj[0,2]
    Bf = torch.tensor(base_feat, dtype=torch.float32, device=device)[:1]
    actions = torch.zeros(1, 3, device=device); actions[:,0]=1.0

    # ============================================================
    # TEST 1: Gradient flow through FK (differentiable)
    # ============================================================
    print("\n" + "="*70)
    print("TEST 1: Differentiable FK — position loss → rotation gradients")
    print("="*70)

    # Forward: model predicts residuals
    logits, mu = model(Ss, Gs, actions, Bf, apply_smooth=True)  # (1, J, K, T, 3)
    # Winner component: pick argmin over K components
    true_res_t = torch.tensor((rot_data[0] - bl[0]).transpose(1,0,2), dtype=torch.float32)  # (J, T, 3)
    diff_to_truth = (mu[0] - true_res_t.unsqueeze(1))**2  # (J, K, T, 3)
    err = diff_to_truth.mean((-2,-1))  # (J, K)
    win = err.argmin(-1)  # (J,)

    # Winner residual: (J, T, 3)
    mu_win = torch.stack([mu[0, j, win[j], :, :] for j in range(J)])  # (J, T, 3)

    # Convert to full rotation using baseline
    bl_t = torch.tensor(bl[0], dtype=torch.float32)  # (T, J, 3)
    pred_rot_t = bl_t + mu_win.permute(1, 0, 2)  # (T, J, 3)
    true_rot_t = torch.tensor(rot_data[0], dtype=torch.float32)  # (T, J, 3)
    trans_t = torch.tensor(trans_data[0], dtype=torch.float32)  # (T, 3)

    # Differentiable FK — need to reshape from (T, J, 3) to (T, 72)
    pred_rot_flat = pred_rot_t.reshape(-1, 72)  # (T, 72)
    true_rot_flat = true_rot_t.reshape(-1, 72)  # (T, 72)
    # pose_body = joints 1-21 (63 dims), root_orient = joint 0 (3 dims)
    pred_pos = fk_smpl_torch(pred_rot_flat[:, 3:66].unsqueeze(0),   # (1, T, 63)
                             pred_rot_flat[:, :3].unsqueeze(0),     # (1, T, 3)
                             trans_t.unsqueeze(0))                   # (1, T, 3)
    true_pos = fk_smpl_torch(true_rot_flat[:, 3:66].unsqueeze(0),
                             true_rot_flat[:, :3].unsqueeze(0),
                             trans_t.unsqueeze(0))

    # Position loss
    pos_loss = ((pred_pos - true_pos)**2).mean()
    pos_loss.backward()

    # Measure gradient norms on pred_rot_t
    # pred_rot_t has requires_grad from the model? No — it was detached.
    # The mu_win comes from model output, which requires_grad.
    # So gradients flow through mu_win → model parameters.
    # Let's check gradient on mu_win directly
    grad_on_mu = mu_win.grad  # (J, T, 3) — gradient of position loss w.r.t. predicted residual
    if grad_on_mu is not None:
        grad_norm_per_joint = grad_on_mu.norm(dim=(1,2))  # (J,) — L2 norm over T,3
    else:
        # pred_rot_t might not retain grad — let's recompute with requires_grad
        print("  mu_win has no grad — recomputing with requires_grad...")
        # Recompute with gradient tracking
        mu_win_check = mu_win.detach().requires_grad_(True)
        pred_rot_check = bl_t.detach() + mu_win_check.permute(1, 0, 2)
        pred_rot_flat_c = pred_rot_check.reshape(-1, 72)
        pred_pos_check = fk_smpl_torch(pred_rot_flat_c[:, 3:66].unsqueeze(0),
                                       pred_rot_flat_c[:, :3].unsqueeze(0),
                                       trans_t.unsqueeze(0))
        pos_loss_check = ((pred_pos_check - true_pos.detach())**2).mean()
        pos_loss_check.backward()
        grad_on_mu = mu_win_check.grad
        grad_norm_per_joint = grad_on_mu.norm(dim=(1,2))

    # Rank joints by gradient norm
    ranking = np.argsort(-grad_norm_per_joint.detach().numpy())

    print(f"\n{'Joint':<14} {'∂L/∂θ norm':>14} {'% of max':>10}")
    print("-"*40)
    max_norm = grad_norm_per_joint.max().item()
    for j in ranking:
        pct = 100 * grad_norm_per_joint[j].item() / max(max_norm, 1e-8)
        marker = " ← FOOT" if j in [7,8,10,11] else ""
        print(f"{JOINT_NAMES[j]:<14} {grad_norm_per_joint[j].item():>14.6f} {pct:>9.1f}%{marker}")

    foot_joints = [7, 8, 10, 11]
    hip_joints = [1, 2]
    foot_norm = grad_norm_per_joint[foot_joints].mean().item()
    hip_norm = grad_norm_per_joint[hip_joints].mean().item()
    print(f"\nFoot avg gradient: {foot_norm:.6f}")
    print(f"Hip avg gradient:  {hip_norm:.6f}")
    print(f"Foot/Hip ratio:    {foot_norm/max(hip_norm,1e-8):.2f}x")

    if foot_norm < 0.01 * max_norm:
        print("\n>>> VERDICT A: Vanishing gradients at feet! FK loss doesn't reach foot rotations. <<<")
        print("    Fix: Switch to differentiable FK in training loop.")
    elif foot_norm > 0.1 * max_norm:
        print("\n>>> VERDICT B: Gradients flow fine. Loss surface issue, not gradient flow. <<<")
        print("    Fix: Per-joint position loss weighting or velocity-space loss.")
    else:
        print("\n>>> VERDICT C: Moderate gradient at feet. Hybrid approach needed. <<<")

    # ============================================================
    # TEST 2: Residual-space vs position-space correlation
    # ============================================================
    print("\n" + "="*70)
    print("TEST 2: Does residual error correlate with position error?")
    print("="*70)

    with torch.no_grad():
        # Residual error per joint
        true_res = (true_rot_t - bl_t).permute(1, 0, 2)  # (J, T, 3)
        res_err = ((mu_win.detach() - true_res)**2).mean(dim=(1,2))  # (J,)
        # Position error per joint
        pos_err = ((pred_pos.detach() - true_pos.detach())**2).mean(dim=(0,1)).sum(-1)  # (J,)
        pos_err_per_joint = pos_err.squeeze().numpy()

        print(f"\n{'Joint':<14} {'Residual MSE':>14} {'Position MSE':>14} {'Correlated?':>12}")
        print("-"*56)
        for j in range(J):
            marker = " ← FOOT" if j in [7,8,10,11] else ""
            print(f"{JOINT_NAMES[j]:<14} {res_err[j].item():>14.6f} {pos_err_per_joint[j]:>14.6f}{marker}")

        # Spearman rank correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(res_err.numpy(), pos_err_per_joint)
        print(f"\nSpearman rank correlation: r={corr:.4f} (p={pval:.4f})")
        if corr < 0.5:
            print(">>> Residual error does NOT correlate with position error! <<<")
            print("    MCL loss in residual space doesn't optimize position accuracy.")
            print("    Fix: Position-space loss is essential.")
        else:
            print(">>> Residual error DOES correlate with position error. <<<")
            print("    Fix: Higher FK loss weight or better residual prediction.")

    print("\nDone.")

if __name__ == "__main__":
    main()
