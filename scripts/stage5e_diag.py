#!/usr/bin/env python3
"""
Per-joint diagnostic: find root cause of sliding.
Trains stage5e model, generates motion, breaks down RMSE per joint.
"""
import sys, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch
from smpl_fk import fk_smpl_full

# Import from stage5e_fixed_v1
from stage5e_fixed_v1 import (load_data, compute_mean_base, baseball_deform,
                               BaseballVertexMDN, train_model, generate_motion)

# SMPL joint names for readability
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
    n_files, n_frames, n_base, n_epochs = 200, 96, 10, 30
    bs, seed = 16, 42
    device = "cpu"

    print("[Diag] Loading data...")
    data, J = load_data(data_dir, n_files, n_frames, seed=seed)

    rng = np.random.default_rng(seed+1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:n_base]]
    train_data = data[perm[n_base:]][:int(0.8*(len(data)-n_base))]
    test_data = data[perm[n_base:]][int(0.8*(len(data)-n_base)):]

    bases = compute_mean_base(base_data[:,:,:72], n_base)
    model = BaseballVertexMDN(n_joints=J, n_timesteps=n_frames).to(device)

    print(f"[Diag] Training {n_epochs} epochs...")
    _, xs, ys = train_model(model, train_data, bases, n_epochs, bs, 1e-3, device, fk_loss_weight=5.0)

    # Test on multiple samples for reliable stats
    n_test = min(5, len(test_data))
    print(f"[Diag] Testing on {n_test} samples...")

    all_pos_err = np.zeros((n_test, n_frames, J))
    all_vel_err = np.zeros((n_test, n_frames-1, J))

    for i in range(n_test):
        test = test_data[i]
        rt = test[:n_frames, :72].reshape(n_frames, J, 3)
        tr = test[:n_frames, 72:75]

        g3d, _ = generate_motion(model, bases, rt[0], rt[-1], tr, xs, ys, device, residual_scale=1.0)
        t3d = fk_smpl_full(rt.reshape(n_frames, 72), tr)

        # Per-joint position error (Euclidean distance)
        pos_err = np.sqrt(((g3d - t3d)**2).sum(axis=-1))  # (T, J)
        all_pos_err[i] = pos_err

        # Per-joint velocity error (temporal derivative)
        g_vel = g3d[1:] - g3d[:-1]  # (T-1, J, 3)
        t_vel = t3d[1:] - t3d[:-1]
        vel_err = np.sqrt(((g_vel - t_vel)**2).sum(axis=-1))  # (T-1, J)
        all_vel_err[i] = vel_err

    # Aggregate across samples: mean per joint over time
    pos_mean = all_pos_err.mean(axis=(0, 1))  # (J,) — mean position error per joint
    vel_mean = all_vel_err.mean(axis=(0, 1))  # (J,) — mean velocity error per joint

    # Rank joints by total error (position + velocity)
    total = pos_mean + vel_mean
    ranking = np.argsort(-total)  # worst first

    print("\n" + "="*70)
    print("PER-JOINT DIAGNOSTIC: Position & Velocity Errors")
    print("="*70)
    print(f"{'Joint':<14} {'Pos RMSE(m)':>12} {'Vel RMSE(m/frame)':>18} {'Total':>10}")
    print("-"*56)

    for rank, j in enumerate(ranking):
        marker = " ← SLIDING?" if j in [7, 8, 10, 11] else ""  # feet/ankles
        print(f"{JOINT_NAMES[j]:<14} {pos_mean[j]:>12.4f} {vel_mean[j]:>18.4f} {total[j]:>10.4f}{marker}")

    # Summary: foot position error vs pelvis
    foot_joints = [7, 8, 10, 11]
    pelvis_j = 0
    print(f"\n--- Summary ---")
    print(f"Pelvis (root) pos RMSE: {pos_mean[pelvis_j]:.4f}m, vel RMSE: {vel_mean[pelvis_j]:.4f}")
    print(f"Feet avg pos RMSE:     {pos_mean[foot_joints].mean():.4f}m, vel RMSE: {vel_mean[foot_joints].mean():.4f}")
    ratio_pos = pos_mean[foot_joints].mean() / max(pos_mean[pelvis_j], 1e-8)
    ratio_vel = vel_mean[foot_joints].mean() / max(vel_mean[pelvis_j], 1e-8)
    print(f"Foot/pelvis ratio:     pos={ratio_pos:.1f}x, vel={ratio_vel:.1f}x")

    if ratio_vel > 1.5:
        print("\n>>> ROOT CAUSE: Feet velocity error >> pelvis. Step speed not matching. <<<")
        print("    Fix: increase FK loss weight on feet, or velocity-matching loss.")
    elif ratio_pos > 1.5:
        print("\n>>> ROOT CAUSE: Feet position error >> pelvis. FK chain accumulation. <<<")
        print("    Fix: per-joint FK weights (feet 5x).")
    else:
        print("\n>>> Errors uniform across joints. Sliding is global, not foot-specific. <<<")
        print("    Fix: overall FK loss weight or amplitude matching.")

    # Per-frame analysis: does error grow over time?
    pos_over_time = all_pos_err.mean(axis=0).mean(axis=-1)  # (T,) — mean over joints & samples
    vel_over_time = all_vel_err.mean(axis=0).mean(axis=-1)  # (T-1,)

    print(f"\n--- Temporal analysis ---")
    print(f"Position error: early={pos_over_time[:20].mean():.4f}m  late={pos_over_time[-20:].mean():.4f}m  drift={pos_over_time[-20:].mean()/max(pos_over_time[:20].mean(),1e-8):.2f}x")
    if pos_over_time[-20:].mean() > 1.5 * pos_over_time[:20].mean():
        print(">>> Error GROWS over time — FK drift. Fix: longer training, stronger FK loss.")

    print("\nDone.")

if __name__ == "__main__":
    main()
