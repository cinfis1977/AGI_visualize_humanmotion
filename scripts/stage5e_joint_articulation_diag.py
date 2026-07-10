#!/usr/bin/env python3
"""Stage5E — Per-joint articulation diagnostic.
Checks EVERY joint's amplitude (gen vs true), not just overall.
"""
import sys, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch

# SMPL 24-joint names (0-indexed, matching FK output order)
JOINT_NAMES_24 = [
    "pelvis",       # 0
    "L_hip",        # 1
    "R_hip",        # 2
    "spine1",       # 3
    "L_knee",       # 4
    "R_knee",       # 5
    "spine2",       # 6
    "L_ankle",      # 7
    "R_ankle",      # 8
    "spine3",       # 9
    "L_foot",       # 10
    "R_foot",       # 11
    "neck",         # 12
    "L_collar",     # 13
    "R_collar",     # 14
    "head",         # 15
    "L_shoulder",   # 16
    "R_shoulder",   # 17
    "L_elbow",      # 18
    "R_elbow",      # 19
    "L_wrist",      # 20
    "R_wrist",      # 21
    "L_hand",       # 22
    "R_hand",       # 23
]

def main():
    device = "cpu"
    import stage5e_per_joint_10base_mdn as s5e
    from smpl_fk import fk_smpl_full

    # Load data (24 joints, 75-d)
    data, n_joints = s5e.load_amass_pose_body(
        "C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz",
        n_files=200, n_frames=96, seed=42)

    rng = np.random.default_rng(43)
    perm = rng.permutation(len(data))

    # Base from 10 walks — 72-d rotation
    base_data = data[perm[:10]]
    rot_base = base_data[:, :, :72].reshape(10, 96, n_joints, 3)
    print("Computing per-joint MEAN bases...")
    bases_rot = s5e.compute_per_joint_mean_bases(
        rot_base.reshape(10, 96, -1), n_base_walks=10)

    # Train — 72-d rotation
    train_data = data[perm[10:170]]
    train_rot = train_data[:, :, :72].reshape(len(train_data), 96, -1)

    model = s5e.PerJointRotationMDN(n_joints=n_joints, n_timesteps=96,
                                     n_components=4, feature_dim=32, hidden_dim=128).to(device)

    print(f"Training ({len(train_data)} samples, 200 epochs, {n_joints} joints)...")
    best_loss, _, x_scaler, y_scaler = s5e.train_model(
        model, train_rot, bases_rot, n_epochs=200, batch_size=32, lr=1e-3, device=device)
    print(f"Training done, best_val={best_loss:.4f}")

    # Test on 5 held-out walks
    test_data = data[perm[170:175]]
    print(f"\n{'='*80}")
    print(f"{'Joint':<14} {'Gen_amp':>8} {'True_amp':>8} {'Ratio':>7} {'RMSE':>8} {'Status'}")
    print(f"{'='*80}")

    all_ratios = []
    all_rmses = []

    for test_idx in range(len(test_data)):
        test = test_data[test_idx]
        rot_test = test[:96, :72].reshape(96, n_joints, 3)  # (T, 24, 3)
        tr_test = test[:96, 72:75]  # (T, 3)

        generated = s5e.generate_motion(model, bases_rot, rot_test[0], rot_test[-1],
                                         tr_test, x_scaler=x_scaler, y_scaler=y_scaler,
                                         n_frames=96, device=device)  # (96, 24, 3)
        true_3d = fk_smpl_full(rot_test.reshape(96, 72), tr_test)  # (96, 24, 3)

        # Per-joint amplitude = std over time, then magnitude of (x,y,z) std
        gen_amp_j = np.std(generated, axis=0)        # (24, 3)
        true_amp_j = np.std(true_3d, axis=0)         # (24, 3)
        gen_amp_mag = np.linalg.norm(gen_amp_j, axis=1)   # (24,)
        true_amp_mag = np.linalg.norm(true_amp_j, axis=1) # (24,)

        ratios = gen_amp_mag / np.maximum(true_amp_mag, 1e-8)
        all_ratios.append(ratios)

        rmse_j = np.sqrt(np.mean((generated - true_3d) ** 2, axis=0))  # (24, 3)
        rmse_mag = np.linalg.norm(rmse_j, axis=1)  # (24,)
        all_rmses.append(rmse_mag)

    # Average across test samples
    avg_ratios = np.mean(all_ratios, axis=0)
    avg_rmses = np.mean(all_rmses, axis=0)

    for j in range(n_joints):
        name = JOINT_NAMES_24[j]
        ratio = avg_ratios[j]
        rmse = avg_rmses[j]

        if ratio > 0.85:
            status = "[OK]"
        elif ratio > 0.60:
            status = "[LOW]"
        else:
            status = "[STIFF!]"

        print(f"{name:<14} {ratio:8.3f} {rmse:8.4f} {status}")

    print(f"{'='*80}")
    good = np.sum(avg_ratios > 0.85)
    low = np.sum((avg_ratios > 0.60) & (avg_ratios <= 0.85))
    stiff = np.sum(avg_ratios <= 0.60)
    print(f"OK: {good}/{n_joints}  LOW: {low}/{n_joints}  STIFF: {stiff}/{n_joints}")
    print(f"Mean ratio: {np.mean(avg_ratios):.3f}")

    if stiff > 0:
        stiff_joints = [JOINT_NAMES_24[j] for j in range(n_joints) if avg_ratios[j] <= 0.60]
        print(f"\nSTIFF JOINTS: {stiff_joints}")
        print("These joints have <60% of true articulation amplitude.")
        print("Consider: more base walks, weaker base, or per-joint articulation scaling.")

if __name__ == "__main__":
    main()
