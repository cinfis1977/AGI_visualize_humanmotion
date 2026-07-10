#!/usr/bin/env python3
"""
Differentiable SMPL Forward Kinematics (FK) in PyTorch.

Layer 2 of the two-layer architecture, in a form that lets a position-space loss
(and a foot-skate loss) backpropagate through FK to the joint rotations. It mirrors
scripts/smpl_fk.py exactly (same KINTREE and template BONE_VECTORS), so bone lengths
remain guaranteed by construction — rotations cannot change them.

Input:  pose_body (..., 63) + root_orient (..., 3) + trans (..., 3)   [any leading dims]
Output: joints (..., 24, 3) in world coordinates

The original numpy scripts/smpl_fk.py is left untouched; import its constants here so
the two implementations can never drift apart.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the exact tree + template bones from the numpy implementation so the torch
# FK is guaranteed to match it (KINTREE[0] is already set to -1 there).
from smpl_fk import BONE_VECTORS as _BONE_VECTORS_NP, KINTREE as _KINTREE_NP, POSE_BODY_TO_SMPL  # noqa: E402

# Python ints for the tree walk; parent index < child index for every SMPL joint,
# so a simple ascending loop processes every parent before its child.
KINTREE = [int(p) for p in _KINTREE_NP]


def axis_angle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues: axis-angle (..., 3) -> rotation matrix (..., 3, 3).

    theta uses a +eps under the sqrt so it is finite and differentiable even at the
    zero-rotation (e.g. the zero-filled hand joints), where it returns identity.
    """
    theta = torch.sqrt((aa * aa).sum(dim=-1, keepdim=True) + 1e-12)  # (..., 1)
    k = aa / theta                                                   # unit axis (..., 3)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz,  ky], dim=-1),
        torch.stack([kz,  zero, -kx], dim=-1),
        torch.stack([-ky,  kx, zero], dim=-1),
    ], dim=-2)                                                       # (..., 3, 3)
    sin_t = torch.sin(theta).unsqueeze(-1)                           # (..., 1, 1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    # Manual 3x3 identity (DirectML-safe: torch.eye unsupported on DML)
    eye = aa.new_zeros(3, 3)
    eye[0, 0] = 1.0; eye[1, 1] = 1.0; eye[2, 2] = 1.0
    return eye + sin_t * K + (1.0 - cos_t) * (K @ K)


def fk_smpl_torch(pose_body: torch.Tensor,
                  root_orient: torch.Tensor,
                  trans: torch.Tensor) -> torch.Tensor:
    """Differentiable FK. See module docstring for shapes."""
    lead = pose_body.shape[:-1]
    N = int(np.prod(lead)) if len(lead) else 1
    device, dtype = pose_body.device, pose_body.dtype

    pb = pose_body.reshape(N, 63)
    ro = root_orient.reshape(N, 3)
    tr = trans.reshape(N, 3)

    # Full 72-d pose: root (3) + 21 body joints (63); hands (joints 22,23) stay zero.
    full = torch.zeros(N, 72, dtype=dtype, device=device)
    full[:, :3] = ro
    full[:, 3:66] = pb
    R = axis_angle_to_rotmat(full.reshape(N, 24, 3))                 # (N, 24, 3, 3)

    bones = torch.as_tensor(_BONE_VECTORS_NP, dtype=dtype, device=device)  # (24, 3)

    world_R = [None] * 24
    world_p = [None] * 24
    world_R[0] = R[:, 0]                                             # (N, 3, 3)
    world_p[0] = tr                                                  # (N, 3)
    for j in range(1, 24):
        p = KINTREE[j]
        # position: parent + parent_world_rotation @ (rest bone offset)
        # Use unsqueeze to avoid addmv (unsupported on DirectML)
        world_p[j] = world_p[p] + (world_R[p] @ bones[j].unsqueeze(-1)).squeeze(-1)  # (N, 3)
        # rotation: parent_world_rotation @ local_rotation
        world_R[j] = torch.matmul(world_R[p], R[:, j])                # (N, 3, 3)

    joints = torch.stack(world_p, dim=1)                            # (N, 24, 3)
    return joints.reshape(*lead, 24, 3)


# ============================================================
# Test: match numpy FK bit-for-bit + gradient flows to pose_body
# ============================================================
if __name__ == "__main__":
    import os
    os.chdir(str(SCRIPT_DIR.parent))  # so smpl_fk's tmp/smpl_neutral.npz path resolves
    from smpl_fk import fk_smpl as fk_np

    base = Path('C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz')
    f = next(iter(base.rglob('Walk*.npz')))
    d = np.load(f, allow_pickle=True)
    idx = np.linspace(0, d['pose_body'].shape[0] - 1, 96, dtype=int)
    pb, ro, tr = d['pose_body'][idx], d['root_orient'][idx], d['trans'][idx]

    j_np = fk_np(pb.reshape(96, -1), ro, tr)                        # (96, 24, 3)
    with torch.no_grad():
        j_t = fk_smpl_torch(
            torch.tensor(pb, dtype=torch.float32),
            torch.tensor(ro, dtype=torch.float32),
            torch.tensor(tr, dtype=torch.float32),
        ).numpy()

    diff = np.abs(j_np - j_t)
    print(f"File: {f.name}")
    print(f"numpy vs torch FK  max_abs_diff = {diff.max():.3e} m   mean = {diff.mean():.3e} m")
    assert diff.max() < 1e-4, "torch FK does not match numpy FK!"
    print("MATCH ok (< 1e-4 m)")

    # gradient check: foot position must be differentiable wrt pose_body
    pb_t = torch.tensor(pb, dtype=torch.float32, requires_grad=True)
    ro_t = torch.tensor(ro, dtype=torch.float32)
    tr_t = torch.tensor(tr, dtype=torch.float32)
    joints = fk_smpl_torch(pb_t, ro_t, tr_t)
    loss = joints[:, 10, :].pow(2).sum()   # SMPL joint 10 = left foot
    loss.backward()
    g = pb_t.grad
    print(f"grad wrt pose_body: finite={torch.isfinite(g).all().item()}  "
          f"nonzero_frac={(g.abs() > 0).float().mean().item():.3f}  norm={g.norm().item():.3f}")
    assert torch.isfinite(g).all() and g.norm() > 0, "gradient did not flow!"
    print("GRAD ok")
