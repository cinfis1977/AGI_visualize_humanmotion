#!/usr/bin/env python3
"""
SMPL Forward Kinematics (FK) converter — no SMPL model dependency.
Uses template bone lengths extracted from SMPL model.

Input:  pose_body (T, 63) + root_orient (T, 3) + trans (T, 3)
Output: joint_positions (T, 24, 3) in world coordinates
"""

import numpy as np

# SMPL kinematic tree: parent index for each of 24 joints
KINTREE = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                     12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int64)
KINTREE[0] = -1  # root has no parent

# Bone vectors: J[child] - J[parent] from SMPL template (T-pose)
# Computed from: tmp/smpl_neutral.npz → J (24, 3) + kintree_table
def _compute_bone_vectors():
    from pathlib import Path
    npz_path = Path(__file__).resolve().parent.parent / 'tmp' / 'smpl_neutral.npz'
    m = np.load(str(npz_path))
    J = m['J']  # (24, 3)
    ktree = m['kintree_table'][0].astype(int)
    ktree[0] = -1
    bones = np.zeros((24, 3), dtype=np.float32)
    for i in range(1, 24):
        p = ktree[i]
        bones[i] = J[i] - J[p]
    return bones

BONE_VECTORS = _compute_bone_vectors()

# Verify
if __name__ == "__main__":
    for i in range(24):
        print(f"  bone {KINTREE[i]}->{i}: {BONE_VECTORS[i]}  len={np.linalg.norm(BONE_VECTORS[i]):.4f}m")

# Index mapping: pose_body[j] (AMASS 21 joints) → which SMPL joint index
# AMASS pose_body covers SMPL joints 1-21 (body only, excluding root=0 and hands=22,23)
POSE_BODY_TO_SMPL = np.arange(1, 22, dtype=int)  # [1,2,...,21]


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to rotation matrix (3,3) using Rodrigues formula."""
    aa = np.asarray(aa, dtype=np.float32).flatten()
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float32)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def fk_smpl(pose_body: np.ndarray, root_orient: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for SMPL skeleton.

    Args:
        pose_body:   (T, 63) or (63,) — body joint axis-angle rotations (21 joints x 3)
        root_orient: (T, 3) or (3,) — root orientation axis-angle
        trans:       (T, 3) or (3,) — root translation

    Returns:
        joints_3d: (T, 24, 3) or (24, 3) — joint positions in world frame
    """
    pose_body = np.atleast_2d(pose_body)
    root_orient = np.atleast_2d(root_orient)
    trans = np.atleast_2d(trans)
    T = pose_body.shape[0]

    # Build full pose: root (3) + 21 body joints (63) → (T, 72)
    # SMPL expects 72-d pose for 24 joints. We zero-fill hands (joints 22,23)
    full_pose = np.zeros((T, 72), dtype=np.float32)
    full_pose[:, :3] = root_orient  # root rotation
    # Map pose_body (21 joints) to full_pose indices 3:66 (3 rotations per joint for joints 1-21)
    full_pose[:, 3:66] = pose_body.reshape(T, 63)
    # Joints 22,23 (hands) remain zero

    joints_world = np.zeros((T, 24, 3), dtype=np.float32)
    rotations = np.zeros((T, 24, 3, 3), dtype=np.float32)  # world-frame rotations

    for t in range(T):
        # Root joint
        joint_positions = np.zeros((24, 3), dtype=np.float32)
        joint_rotations = np.zeros((24, 3, 3), dtype=np.float32)

        # Root (joint 0)
        rot0 = axis_angle_to_rotmat(full_pose[t, :3])
        joint_positions[0] = trans[t]
        joint_rotations[0] = rot0

        # Walk kinematic tree (topological order: parent always processed before child)
        # KINTREE is sorted by joint index which respects hierarchy for SMPL
        for j in range(1, 24):
            p = KINTREE[j]
            if p < 0:
                continue
            # Bone vector from parent to this joint (already J[child]-J[parent])
            bone = BONE_VECTORS[j]
            # Transform bone by parent's world rotation
            bone_world = joint_rotations[p] @ bone
            # Joint position in world
            joint_positions[j] = joint_positions[p] + bone_world
            # Joint rotation: parent_rotation @ local_rotation
            local_rot = axis_angle_to_rotmat(full_pose[t, j*3:(j+1)*3])
            joint_rotations[j] = joint_rotations[p] @ local_rot

        joints_world[t] = joint_positions

    return joints_world.squeeze()


def fk_smpl_full(full_pose: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """FK with full 72-d pose (24 joints × 3).

    Args:
        full_pose: (T, 72) or (72,) — [root(3), body_j1..j21(63), L_hand(3), R_hand(3)]
        trans:     (T, 3) or (3,) — root translation

    Returns:
        joints_3d: (T, 24, 3) or (24, 3) — ALL 24 joints including hands
    """
    full_pose = np.atleast_2d(full_pose)
    trans = np.atleast_2d(trans)
    T = full_pose.shape[0]
    joints_world = np.zeros((T, 24, 3), dtype=np.float32)
    for t in range(T):
        joint_positions = np.zeros((24, 3), dtype=np.float32)
        joint_rotations = np.zeros((24, 3, 3), dtype=np.float32)
        rot0 = axis_angle_to_rotmat(full_pose[t, :3])
        joint_positions[0] = trans[t]
        joint_rotations[0] = rot0
        for j in range(1, 24):
            p = KINTREE[j]
            if p < 0:
                continue
            bone = BONE_VECTORS[j]
            bone_world = joint_rotations[p] @ bone
            joint_positions[j] = joint_positions[p] + bone_world
            local_rot = axis_angle_to_rotmat(full_pose[t, j * 3:(j + 1) * 3])
            joint_rotations[j] = joint_rotations[p] @ local_rot
        joints_world[t] = joint_positions
    return joints_world.squeeze()


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    # Load a real AMASS walk file and convert
    from pathlib import Path
    base = Path('C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz')
    walk_files = sorted(base.rglob('Walk*.npz'))
    f = walk_files[0]
    d = np.load(f, allow_pickle=True)

    pb = d['pose_body'][:100]  # first 100 frames
    ro = d['root_orient'][:100]
    tr = d['trans'][:100]

    joints = fk_smpl(pb, ro, tr)
    print(f"Pose shape: {pb.shape} → Joints shape: {joints.shape}")

    # Check ranges
    for j in range(24):
        j_pos = joints[:, j, :]
        print(f"  joint {j:2d}: x=[{j_pos[:,0].min():.2f},{j_pos[:,0].max():.2f}] "
              f"y=[{j_pos[:,1].min():.2f},{j_pos[:,1].max():.2f}] "
              f"z=[{j_pos[:,2].min():.2f},{j_pos[:,2].max():.2f}]")

    # Check bone lengths are preserved
    print("\nBone length preservation (should be constant):")
    for j in range(1, 24):
        p = KINTREE[j]
        if p < 0:
            continue
        lengths = np.linalg.norm(joints[:, j, :] - joints[:, p, :], axis=1)
        print(f"  bone {p}->{j}: mean={lengths.mean():.4f}m std={lengths.std():.4f}m")
