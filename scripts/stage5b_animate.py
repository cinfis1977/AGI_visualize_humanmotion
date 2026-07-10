#!/usr/bin/env python3
"""Quick walk animation from Stage5B one-shot model."""
import sys, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# SMPL kinematic tree for stick figure (parent->child connections)
BONES = [
    (0,1), (0,2), (0,3),         # pelvis → hips, spine
    (1,4), (4,7), (7,10),        # left leg
    (2,5), (5,8), (8,11),        # right leg
    (3,6), (6,9), (9,12),        # spine → neck
    (9,13), (13,16), (16,18), (18,20),  # left arm
    (9,14), (14,17), (17,19), (19,21),  # right arm
    (12,15),                      # neck → head
    (1,3), (2,3),                 # connect hips to spine (pelvis proxy)
]
# Use only 21 body joints (indices 1-21 mapped to 0-20 in our data)
# Our data has joints 1-21, so reindex: bone references use SMPL indices
# Map SMPL joint indices to our 21-joint index (our idx 0 = SMPL joint 1)
def our_bone(bone_smpl):
    a, b = bone_smpl
    if a == 0 or b == 0:
        return None  # root (pelvis) not in our 21 joints
    if a > 21 or b > 21:
        return None
    return (a-1, b-1)

BONES_OUR = [b for b in [our_bone(b) for b in BONES] if b is not None]

def main():
    import torch
    device = "cpu"

    # Quick train
    import stage5b_amass_walk_mdn as s5b
    
    # Use ONLY straight walks (filter at load time)
    from smpl_fk import fk_smpl
    all_data, n_joints = s5b.load_amass_walk_files(
        "C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz",
        n_files=200, n_frames=96, seed=42
    )
    
    # Filter: keep walks with >1m forward x motion AND no turn/support keywords
    good = []
    data_3d = all_data.reshape(len(all_data), 96, n_joints, 3)
    pelvis_x_range = data_3d[:, :, 0, 0].max(axis=1) - data_3d[:, :, 0, 0].min(axis=1)
    for i in range(len(all_data)):
        if pelvis_x_range[i] > 1.5:
            good.append(all_data[i])
    data = np.stack(good) if good else all_data
    print(f"Straight walks: {len(data)} samples (x_range > 1.5m)")

    # Base from 10 samples
    rng = np.random.default_rng(43)
    perm = rng.permutation(len(data))
    base_data = data[perm[:10]]
    train_data = data[perm[10:40]]
    test_data = data[perm[40:]]
    bases = s5b.compute_bases(base_data)
    
    model = s5b.PerJointOneShotMDN(n_joints=n_joints, n_timesteps=96, n_components=4,
                                    feature_dim=32, hidden_dim=128).to(device)
    
    print("Training...")
    s5b.train_model_one_shot(model, train_data, bases, n_epochs=100, batch_size=16, lr=1e-3, device=device)
    
    # Generate from a test sample
    test_sample = test_data[0].reshape(96, n_joints, 3)
    start = test_sample[0]
    goal = test_sample[-1]
    generated = s5b.generate_motion_one_shot(model, bases, start, goal, n_frames=96, device=device)
    
    print(f"Generated: {generated.shape}")
    
    # Animation
    out_path = Path("outputs/stage5b_walk_anim.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    true_sample = test_sample  # actual for comparison
    
    def update(frame):
        ax.cla()
        joints = generated[frame]  # (21, 3)
        
        # Draw bones
        for a, b in BONES_OUR:
            ax.plot([joints[a, 0], joints[b, 0]],
                    [joints[a, 1], joints[b, 1]],
                    [joints[a, 2], joints[b, 2]], 'b-', linewidth=2, alpha=0.9)
        
        # Draw joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', s=20)
        
        # Draw true as faint overlay
        true_joints = true_sample[frame]
        for a, b in BONES_OUR:
            ax.plot([true_joints[a, 0], true_joints[b, 0]],
                    [true_joints[a, 1], true_joints[b, 1]],
                    [true_joints[a, 2], true_joints[b, 2]], 'r-', linewidth=1, alpha=0.3)
        
        ax.set_xlim(-1.5, 2.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.5, 2.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Walk Frame {frame+1}/96\nBlue=Generated  Red=True')
    
    anim = FuncAnimation(fig, update, frames=96, interval=120)  # slower: 120ms per frame
    anim.save(str(out_path), writer=PillowWriter(fps=8))
    plt.close()
    print(f"Saved: {out_path}")
    print(f"RMSE: {np.sqrt(np.mean((generated - true_sample)**2)):.3f}m")

if __name__ == "__main__":
    main()
