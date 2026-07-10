#!/usr/bin/env python3
"""Scaler sanity check — bypasses neural network entirely."""
import sys, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch
from smpl_fk import fk_smpl_full
from stage5e_fixed_v1 import load_data, compute_mean_base, baseball_deform

data, J = load_data('C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz', 200, 96, seed=42)
rng = np.random.default_rng(43)
perm = rng.permutation(len(data))
base_data = data[perm[:10]]
train_data = data[perm[10:]][:int(0.8*(len(data)-10))]
bases = compute_mean_base(base_data[:,:,:72], 10)

T = 96
sample_raw = train_data[0:1]
rot_raw = sample_raw[:,:,:72].reshape(1, T, J, 3)
trans_raw = sample_raw[:,:,72:75]

# Reconstruct baseline
bl_np = np.zeros((1, T, J, 3), dtype=np.float32)
for j in range(J):
    bl_np[0,:,j] = baseball_deform(bases[j], rot_raw[0,0,j], rot_raw[0,-1,j])
BL = torch.tensor(bl_np, dtype=torch.float32)
R = torch.tensor(rot_raw, dtype=torch.float32)

# Residual (raw radians, NO scaler — this is what stage5e_fixed_v1 does)
res_true = R.permute(0,2,1,3) - BL.permute(0,2,1,3)  # (1,J,T,3)

# Test 1: No-scale (direct raw radians → FK)
res_np_direct = res_true.permute(0,2,1,3).cpu().numpy()  # (1,T,J,3)
rot_direct = (bl_np + res_np_direct).squeeze(0).reshape(T, J*3).astype(np.float32)
joints_true = fk_smpl_full(rot_raw.squeeze(0).reshape(T, J*3).astype(np.float32), trans_raw.squeeze(0).astype(np.float32))
joints_direct = fk_smpl_full(rot_direct, trans_raw.squeeze(0).astype(np.float32))
print(f"No-scale direct roundtrip RMSE: {np.sqrt(np.mean((joints_true-joints_direct)**2)):.8f}m")

# Test 2: With global Scaler (like stage5p v4) — operates on (J, T*3) features
from stage5p_clean_baseball_clean_natural_v4 import Scaler

# Residual per joint: (1, J, T, 3)
res_joint = res_true  # (1, J, T, 3)
# Flatten to (J, T*3) = (J, 288) for scaler (per-joint features)
res_flat = res_joint.squeeze(0).cpu().numpy().reshape(J, -1)  # (24, 288)
# Fit global scaler on all joints
scaler = Scaler().fit(res_flat)
# Transform → inverse roundtrip
scaled = scaler.transform(res_flat)  # (24, 288)
inv_flat = scaler.inverse(scaled)    # (24, 288)
# Reshape back to (1, J, T, 3)
inv = inv_flat.reshape(J, T, 3)[None, ...]  # (1, 24, 96, 3)

# Convert back to (1, T, J, 3) for FK
rot_inv = (bl_np + inv.transpose(0, 2, 1, 3)).squeeze(0).reshape(T, J*3).astype(np.float32)
joints_inv = fk_smpl_full(rot_inv, trans_raw.squeeze(0).astype(np.float32))
rmse_scaled = np.sqrt(np.mean((joints_true-joints_inv)**2))
print(f"Global Scaler roundtrip RMSE: {rmse_scaled:.8f}m")
if rmse_scaled > 1e-4:
    print("[WARNING] Global scaler introduces amplitude loss!")
else:
    print("[OK] Global scaler preserves amplitude within tolerance")
