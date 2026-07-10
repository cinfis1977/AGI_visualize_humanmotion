#!/usr/bin/env python3
"""Animation for Stage5E — per-joint mean base (10 walks)."""
import sys, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

# SMPL skeleton bones (parent->child pairs, 1-indexed SMPL joint IDs)
BONES = [(0,1),(0,2),(0,3),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
         (3,6),(6,9),(9,12),(9,13),(13,16),(16,18),(18,20),
         (9,14),(14,17),(17,19),(19,21),(12,15),(1,3),(2,3)]
def our_bone(b): a,b=b; return None if a==0 or b==0 or a>21 or b>21 else (a-1,b-1)
BONES_OUR = [b for b in [our_bone(b) for b in BONES] if b is not None]

def main():
    device = "cpu"
    import stage5e_per_joint_10base_mdn as s5e
    from smpl_fk import fk_smpl_full

    # Load data (now 24 joints)
    data, n_joints = s5e.load_amass_pose_body(
        "C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz",
        n_files=200, n_frames=96, seed=42)
    print(f"Loaded {len(data)} walks ({n_joints} joints)")

    # Pick a good walk with visible leg motion for test
    good_walk_files = ['walking_medium01_stageii.npz', 'walk_6m_straight_line04_stageii.npz']
    good_data = None
    for fname in good_walk_files:
        for f in Path('C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz').rglob(fname):
            d = np.load(f, allow_pickle=True)
            idx = np.linspace(0, d['pose_body'].shape[0]-1, 96, dtype=int)
            ro = d['root_orient'][idx]
            pb = d['pose_body'][idx]
            tr = d['trans'][idx]
            hand_z = np.zeros((96, 6), dtype=np.float32)
            full_rot = np.concatenate([ro, pb, hand_z], axis=1)
            good_data = np.concatenate([full_rot, tr], axis=1)
            print(f"Test walk: {f.name}")
            break
        if good_data is not None:
            break

    # Split: 10 base walks, rest for train
    rng = np.random.default_rng(43)
    perm = rng.permutation(len(data))

    base_data = data[perm[:10]]
    rot_base = base_data[:, :, :72].reshape(10, 96, n_joints, 3)

    # Stage5E: MEAN base from 10 walks (NOT medoid!)
    print("Computing per-joint MEAN bases from 10 walks...")
    bases_rot = s5e.compute_per_joint_mean_bases(
        rot_base.reshape(10, 96, -1), n_base_walks=10)

    # Train split
    train_data = data[perm[10:170]]
    train_rot = train_data[:, :, :72].reshape(len(train_data), 96, -1)

    # Model
    model = s5e.PerJointRotationMDN(n_joints=n_joints, n_timesteps=96, n_components=4,
                                     feature_dim=32, hidden_dim=128).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("Training (MEAN base -> large residuals -> full articulation)...")
    t0 = __import__('time').time()
    best_loss, _, x_scaler, y_scaler, base_pos_3d = s5e.train_model(
        model, train_rot, bases_rot, n_epochs=150, batch_size=32, lr=1e-3, device=device)
    print(f"Training: {__import__('time').time()-t0:.1f}s  best={best_loss:.4f}")

    # Generate from test walk
    if good_data is not None:
        test = good_data
    else:
        test = data[perm[170]]
    rot_test = test[:96, :72].reshape(96, n_joints, 3)
    tr_test = test[:96, 72:75]

    test_pos = fk_smpl_full(rot_test.reshape(96, 72), tr_test)  # (96, 24, 3)
    generated = s5e.generate_motion(model, base_pos_3d, test_pos[0], test_pos[-1],
                                     x_scaler=x_scaler, y_scaler=y_scaler,
                                     n_frames=96, device=device)
    true_3d = test_pos

    # ---- Animation ----
    out = Path("outputs/stage5e_walk_anim.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    HALF = 1.2  # metres, camera window half-width

    def update(frame):
        ax.cla()
        j = generated[frame]      # generated (24, 3)
        tj = true_3d[frame]       # true (24, 3)

        # Draw bones
        for a, b in BONES_OUR:
            ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], [j[a, 2], j[b, 2]],
                    'b-', lw=2.5, alpha=0.9)
            ax.plot([tj[a, 0], tj[b, 0]], [tj[a, 1], tj[b, 1]], [tj[a, 2], tj[b, 2]],
                    'r-', lw=1.5, alpha=0.35)

        # Draw joints
        ax.scatter(j[:, 0], j[:, 1], j[:, 2], c='blue', s=25, alpha=0.9)
        ax.scatter(tj[:, 0], tj[:, 1], tj[:, 2], c='red', s=15, alpha=0.3)

        # Camera follows walker
        c = tj.mean(axis=0)
        ax.set_xlim(c[0] - HALF, c[0] + HALF)
        ax.set_ylim(c[1] - HALF, c[1] + HALF)
        ax.set_zlim(c[2] - HALF, c[2] + HALF)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (side)')
        ax.set_zlabel('Z (up)')
        ax.set_title(f'Stage5E Walk  Frame {frame+1}/96  Blue=Gen  Red=True',
                     fontsize=12)

    anim = FuncAnimation(fig, update, frames=96, interval=120)
    anim.save(str(out), writer=PillowWriter(fps=8))
    plt.close()

    # Metrics
    rmse = np.sqrt(np.mean((generated - true_3d) ** 2))
    gen_amp = np.std(generated, axis=0).mean()
    true_amp = np.std(true_3d, axis=0).mean()
    art_ratio = gen_amp / max(true_amp, 1e-8)

    print(f"\nSaved: {out}")
    print(f"RMSE: {rmse:.4f}m")
    print(f"Articulation: gen={gen_amp:.4f}  true={true_amp:.4f}  ratio={art_ratio:.2f}")
    if art_ratio > 0.7:
        print("[OK] Full articulation recovered!")
    else:
        print("[WARN] Articulation still low")


if __name__ == "__main__":
    main()
