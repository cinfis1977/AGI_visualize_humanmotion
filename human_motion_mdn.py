#!/usr/bin/env python3
"""
human_motion_mdn.py — Human Motion MDN (ported from baseball trajectory model)
================================================================================
Core architecture: JointPathMDN from stage5_true_mdn_joint_imagination.py
Adapted for full-body human motion trajectory imagination.

Architecture (identical to baseball model):
  - 3-layer MLP trunk (GELU, dropout 0.05)
  - 3 heads: logits (K), mu (K*D), log_sigma (K)
  - Isotropic Gaussian MDN with NLL + MSE surrogate loss
  - K components learn different "motion styles" (walk, run, jump, etc.)

Key differences from baseball:
  - Input:  action one-hot + start_pose + goal_pose + betas + base features
  - Output: 64 frames × N_joints × 3D = full-body trajectory
  - Larger hidden dim (512 vs 256), more components (16 vs 12)

Usage:
  python human_motion_mdn.py
  python human_motion_mdn.py --epochs 300 --n-components 16 --hidden-dim 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Torch ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    print("[FATAL] PyTorch required. Install: pip install torch")
    sys.exit(1)

Array = np.ndarray
Tensor = torch.Tensor

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent  # script'in bulunduğu dizin
DATA_DIR = PROJECT_ROOT / "data"
MINI_DIR = DATA_DIR / "mini_amass"
OUTPUT_DIR = PROJECT_ROOT / "human_motion_mdn_out"

# SMPL-H joint mapping (subset of key joints for trajectory prediction)
KEY_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,   "right_hip": 2,
    "spine1": 3,     "spine2": 6, "spine3": 9,
    "left_knee": 4,  "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
    "left_foot": 10, "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,  "right_shoulder": 17,
    "left_elbow": 18,     "right_elbow": 19,
    "left_wrist": 20,     "right_wrist": 21,
}
N_KEY_JOINTS = len(KEY_JOINTS)  # 19 joints
JOINT_DIM = 3
FRAMES = 32  # reduced from 64 for memory efficiency
POSE_DIM = 63  # body pose only (joints 1-21 × 3), excluding hands and root orient

# Action categories in the dataset
ACTION_CATEGORIES = [
    "walk", "run", "jog", "jump", "sit", "stand",
    "bend", "crouch", "squat", "turn", "fall",
    "kick", "throw", "crawl",
]
N_ACTIONS = len(ACTION_CATEGORIES)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CATEGORIES)}

# Default hyperparameters (from baseball model)
DEFAULT_N_COMPONENTS = 16
DEFAULT_HIDDEN_DIM = 512
DEFAULT_N_HIDDEN = 3
DEFAULT_DROPOUT = 0.05
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_FRAC = 0.15
DEFAULT_PATIENCE = 50
DEFAULT_MSE_AUX = 0.05
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SIGMA_PENALTY = 0.0
DEFAULT_TOP_K = 3


# ══════════════════════════════════════════════════════════════
# StandardScaler (pure numpy, ported from baseball model)
# ══════════════════════════════════════════════════════════════

@dataclass
class StandardScalerNP:
    mean: Optional[Array] = None
    std: Optional[Array] = None
    eps: float = 1e-8

    def fit(self, X: Array):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + self.eps

    def transform(self, X: Array) -> Array:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: Array) -> Array:
        return X * self.std + self.mean


# ══════════════════════════════════════════════════════════════
# MDN Network (ported from stage5_true_mdn_joint_imagination.py)
# ══════════════════════════════════════════════════════════════

class JointPathMDN(nn.Module):
    """Mixture Density Network for trajectory imagination.
    
    Identical architecture to the baseball model's JointPathMDN.
    """
    
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        n_components: int = DEFAULT_N_COMPONENTS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_hidden: int = DEFAULT_N_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        K = n_components
        D = target_dim

        # Trunk
        layers = []
        in_dim = input_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Heads
        self.logits_head = nn.Linear(hidden_dim, K)
        self.mu_head = nn.Linear(hidden_dim, K * D)
        self.log_sigma_head = nn.Linear(hidden_dim, K)

        self._init_weights()

    def _init_weights(self):
        # Conservative initialization (from baseball model)
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.bias)
        # logits head: default Kaiming is fine

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(x)
        logits = self.logits_head(h)                          # (B, K)
        mu = self.mu_head(h).view(-1, self.n_components, self.target_dim)  # (B, K, D)
        log_sigma = torch.clamp(self.log_sigma_head(h), -4.0, 3.0)  # (B, K)
        return logits, mu, log_sigma


# ══════════════════════════════════════════════════════════════
# MDN Loss (ported from baseball model)
# ══════════════════════════════════════════════════════════════

def mdn_nll_isotropic(logits: Tensor, mu: Tensor, log_sigma: Tensor, y: Tensor) -> Tensor:
    """Isotropic Gaussian Mixture NLL."""
    K = logits.size(1)
    D = mu.size(2)
    log_pi = F.log_softmax(logits, dim=1)                     # (B, K)
    sigma = torch.exp(log_sigma) + 1e-6                       # (B, K)

    # (B, K, D) -> compute squared Mahalanobis-ish term
    diff = y.unsqueeze(1) - mu                                # (B, 1, D) - (B, K, D) = (B, K, D)
    sq_norm = (diff * diff).sum(dim=2)                        # (B, K)
    log_prob = (
        -0.5 * D * math.log(2 * math.pi)
        - D * torch.log(sigma)
        - 0.5 * sq_norm / (sigma * sigma + 1e-8)
    )                                                          # (B, K)

    log_mix = log_pi + log_prob                               # (B, K)
    log_sum = torch.logsumexp(log_mix, dim=1)                 # (B,)
    return -log_sum.mean()


def mixture_mse_surrogate(logits: Tensor, mu: Tensor, y: Tensor) -> Tensor:
    """Weighted-mean MSE auxiliary loss."""
    weights = F.softmax(logits, dim=1)                         # (B, K)
    mu_weighted = (weights.unsqueeze(-1) * mu).sum(dim=1)     # (B, D)
    return F.mse_loss(mu_weighted, y)


def mdn_total_loss(logits, mu, log_sigma, y, mse_weight=DEFAULT_MSE_AUX):
    return mdn_nll_isotropic(logits, mu, log_sigma, y) + mse_weight * mixture_mse_surrogate(logits, mu, y)


# ══════════════════════════════════════════════════════════════
# SMPL Pose -> Joint Positions (simplified: use nearest neighbors)
# ══════════════════════════════════════════════════════════════

def poses_to_keypoint_positions(poses: Array) -> Array:
    """Extract approximate 3D positions of key joints from SMPL-H pose params.
    
    Without the full SMPL model, we use a simplified approach:
    - Extract root translation and orientation from poses
    - Reconstruct a rough kinematic chain from the pose angles
    
    For now, returns a placeholder that works with the data format.
    In production, use the SMPL-H body model via smplx library.
    
    Args:
        poses: (T, 156) SMPL-H pose parameters
    
    Returns:
        (T, N_KEY_JOINTS, 3) approximate 3D joint positions
    """
    # Simplified: use first 3 params as root orientation,
    # params 3:66 as body pose (21 joints × 3),
    # reconstruct chain roughly
    
    T = poses.shape[0]
    # For a quick implementation, we use a simple learned linear mapping
    # In production, replace with SMPL forward pass
    
    # Use the pose body parameters (66-d) as proxy for joint angles
    # and reconstruct approximate positions via a simple kinematic chain
    
    # This is a placeholder — real implementation needs SMPL model
    # For the data we have (SMPL-H params), we store poses directly
    # and defer joint computation to evaluation time
    
    # Return a dummy that preserves the structure
    kp = np.zeros((T, N_KEY_JOINTS, 3), dtype=np.float32)
    # Fill with pose body params reshaped as a rough approximation
    body_pose = poses[:, 3:66]  # (T, 63)
    # Simple: reshape to (T, 21, 3) and take first N_KEY_JOINTS
    body_reshaped = body_pose.reshape(T, 21, 3)
    kp[:, :min(N_KEY_JOINTS, 21), :] = body_reshaped[:, :N_KEY_JOINTS, :]
    return kp


# ══════════════════════════════════════════════════════════════
# Data Loading & Feature Engineering
# ══════════════════════════════════════════════════════════════

def load_mini_amass() -> List[dict]:
    """Load all samples from mini_amass dataset."""
    with open(MINI_DIR / "index.json") as f:
        index = json.load(f)
    
    samples = []
    for s in index["samples"]:
        npz_path = MINI_DIR / s["file"]
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=True))
            data["file"] = s["file"]
            samples.append(data)
    return samples


def build_feature_vector(
    sample: dict,
    action_idx: int,
    base_mean_poses: Optional[Array] = None,
) -> Array:
    """Build input feature vector for the MDN.
    
    Features:
    - Action one-hot: N_ACTIONS (14)
    - Start pose (body only, 63-d): 63
    - Goal pose (body only, 63-d): 63
    - Delta pose (goal - start): 63
    - Body shape (betas): 16
    - Duration info: 2 (fps, duration_sec)
    - Base path descriptors: 5
    
    Total: 14 + 63 + 63 + 63 + 16 + 2 + 5 = 226
    """
    poses_full = np.array(sample["poses"])  # (T, 156) — copy for safety
    # Extract body pose only: indices 3:66 (63-d)
    poses_body = poses_full[:, 3:66].astype(np.float32)  # (T, 63)
    
    # Resample to FRAMES
    T_orig = poses_body.shape[0]
    if T_orig != FRAMES:
        old_t = np.linspace(0, 1, T_orig)
        new_t = np.linspace(0, 1, FRAMES)
        poses_body = np.array([
            np.interp(new_t, old_t, poses_body[:, i])
            for i in range(poses_body.shape[1])
        ]).T.astype(np.float32)
    
    betas = np.array(sample.get("betas", np.zeros(16, dtype=np.float32)), dtype=np.float32)
    fps = float(sample.get("fps", 30.0))
    duration = float(sample.get("duration_sec", 2.0))
    
    start_pose = poses_body[0]       # (63,)
    goal_pose = poses_body[-1]       # (63,)
    delta_pose = goal_pose - start_pose  # (63,)
    
    # One-hot action
    action_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    if 0 <= action_idx < N_ACTIONS:
        action_oh[action_idx] = 1.0
    
    # Base path descriptors
    pose_diffs = np.diff(poses_body, axis=0)
    path_len = float(np.mean(np.linalg.norm(pose_diffs, axis=1)))
    curv = float(np.mean(np.linalg.norm(np.diff(pose_diffs, axis=0), axis=1))) if len(pose_diffs) > 1 else 0.0
    mean_speed = path_len / max(duration, 0.01)
    max_dev = float(np.max(np.abs(poses_body - np.mean(poses_body, axis=0))))
    final_drop = float(np.linalg.norm(poses_body[-1] - poses_body[-2]))
    base_features = np.array([path_len, curv, mean_speed, max_dev, final_drop], dtype=np.float32)
    
    dur_features = np.array([fps / 120.0, duration / 10.0], dtype=np.float32)
    
    x = np.concatenate([
        action_oh,       # 14
        start_pose,      # 63
        goal_pose,       # 63
        delta_pose,      # 63
        betas,           # 16
        dur_features,    # 2
        base_features,   # 5
    ])  # Total: 226
    
    return x.astype(np.float32)


def build_target_vector(sample: dict) -> Array:
    """Build target (Y): residual body pose from start, flattened.
    
    Uses body pose only (63-d), FRAMES frames.
    First and last frames zero-pinned.
    
    Shape: (FRAMES * 63,) = (2016,)
    """
    poses_full = np.array(sample["poses"], dtype=np.float32)  # (T, 156) — copy for safety
    poses_body = poses_full[:, 3:66]                              # (T, 63)
    
    # Resample to FRAMES
    T_orig = poses_body.shape[0]
    if T_orig != FRAMES:
        old_t = np.linspace(0, 1, T_orig)
        new_t = np.linspace(0, 1, FRAMES)
        poses_body = np.array([
            np.interp(new_t, old_t, poses_body[:, i])
            for i in range(poses_body.shape[1])
        ]).T.astype(np.float32)
    
    start_pose = poses_body[0].copy()
    residual = poses_body - start_pose
    
    # Pin endpoints
    residual[0] = 0.0
    residual[-1] = 0.0
    
    return residual.reshape(-1).astype(np.float32)


def prepare_dataset(
    samples: List[dict],
    max_per_action: Optional[int] = None,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = 42,
):
    """Build X, Y arrays from mini_amass samples."""
    rng = np.random.default_rng(seed)
    
    X_list, Y_list = [], []
    action_counts = defaultdict(int)
    
    for sample in samples:
        action = str(sample.get("action_label", "other"))
        if action not in ACTION_TO_IDX:
            continue
        
        action_idx = ACTION_TO_IDX[action]
        
        if max_per_action and action_counts[action] >= max_per_action:
            continue
        
        x = build_feature_vector(sample, action_idx)
        y = build_target_vector(sample)
        
        X_list.append(x)
        Y_list.append(y)
        action_counts[action] += 1
    
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    
    # Shuffle and split
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    
    print(f"Dataset: {len(X)} total, {len(train_idx)} train, {len(val_idx)} val")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    
    return (
        X[train_idx], Y[train_idx],
        X[val_idx], Y[val_idx],
    )


# ══════════════════════════════════════════════════════════════
# Training (ported from baseball model)
# ══════════════════════════════════════════════════════════════

def train_mdn(
    model: JointPathMDN,
    X_train: Array, Y_train: Array,
    X_val: Array, Y_val: Array,
    x_scaler: StandardScalerNP,
    y_scaler: StandardScalerNP,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    device: str = "cpu",
    out_dir: Optional[Path] = None,
) -> dict:
    """Train the MDN model."""
    
    # Standardize
    X_train_s = x_scaler.transform(X_train)
    Y_train_s = y_scaler.transform(Y_train)
    X_val_s = x_scaler.transform(X_val)
    Y_val_s = y_scaler.transform(Y_val)
    
    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(Y_train_s, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(Y_val_s, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_nll": [], "val_nll": []}
    
    print(f"\nTraining on {device} | {epochs} epochs | batch={batch_size} | lr={lr}")
    print(f"  Input dim: {model.input_dim} | Target dim: {model.target_dim}")
    print(f"  Components: {model.n_components} | Hidden: {model.hidden_dim}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    t0 = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_nll_sum = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, mu, log_sigma = model(xb)
            loss = mdn_total_loss(logits, mu, log_sigma, yb)
            nll = mdn_nll_isotropic(logits, mu, log_sigma, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_nll_sum += nll.item() * xb.size(0)
        
        train_loss = train_loss_sum / len(train_ds)
        train_nll = train_nll_sum / len(train_ds)
        
        # Val
        model.eval()
        val_loss_sum = 0.0
        val_nll_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, mu, log_sigma = model(xb)
                val_loss_sum += mdn_total_loss(logits, mu, log_sigma, yb).item() * xb.size(0)
                val_nll_sum += mdn_nll_isotropic(logits, mu, log_sigma, yb).item() * xb.size(0)
        
        val_loss = val_loss_sum / len(val_ds)
        val_nll = val_nll_sum / len(val_ds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        
        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"Train NLL: {train_nll:.4f} | Val NLL: {val_nll:.4f} | "
                  f"Best: {best_val_loss:.4f} | {elapsed:.0f}s")
        
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s | Best val loss: {best_val_loss:.4f}")
    
    # Save
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "x_scaler_mean": x_scaler.mean,
            "x_scaler_std": x_scaler.std,
            "y_scaler_mean": y_scaler.mean,
            "y_scaler_std": y_scaler.std,
            "history": history,
            "config": {
                "input_dim": model.input_dim,
                "target_dim": model.target_dim,
                "n_components": model.n_components,
                "hidden_dim": model.hidden_dim,
            },
        }, out_dir / "human_motion_mdn.pt")
        print(f"Saved: {out_dir / 'human_motion_mdn.pt'}")
    
    return history


# ══════════════════════════════════════════════════════════════
# Prediction & Evaluation
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_motion(
    model: JointPathMDN,
    x: Array,
    x_scaler: StandardScalerNP,
    y_scaler: StandardScalerNP,
    temperature: float = DEFAULT_TEMPERATURE,
    sigma_penalty: float = DEFAULT_SIGMA_PENALTY,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
) -> Array:
    """Generate a motion trajectory from input features.
    
    Uses weighted component combination (like baseball model).
    """
    model.eval()
    model = model.to(device)
    
    x_s = x_scaler.transform(x.reshape(1, -1))
    x_t = torch.tensor(x_s, dtype=torch.float32, device=device)
    
    logits, mu, log_sigma = model(x_t)
    
    # Component scoring with temperature and sigma penalty
    scores = logits / temperature - sigma_penalty * log_sigma  # (1, K)
    weights = F.softmax(scores, dim=1)                          # (1, K)
    
    # Weighted combination
    mu_weighted = (weights.unsqueeze(-1) * mu).sum(dim=1)       # (1, D)
    
    y_pred_s = mu_weighted.cpu().numpy().squeeze(0)
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(1, -1)).squeeze(0)
    
    return y_pred


def trajectory_rmse(y_pred: Array, y_true: Array) -> float:
    """Root mean squared error per frame."""
    D = y_pred.shape[-1] if y_pred.ndim > 1 else 1
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def evaluate_model(
    model: JointPathMDN,
    X_test: Array, Y_test: Array,
    x_scaler: StandardScalerNP,
    y_scaler: StandardScalerNP,
    device: str = "cpu",
) -> dict:
    """Evaluate MDN on test set."""
    model.eval()
    model = model.to(device)
    
    X_s = x_scaler.transform(X_test)
    Y_s = y_scaler.transform(Y_test)
    
    test_ds = TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(Y_s, dtype=torch.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    total_nll = 0.0
    all_rmse = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, mu, log_sigma = model(xb)
            total_nll += mdn_nll_isotropic(logits, mu, log_sigma, yb).item() * xb.size(0)
            
            # Weighted-mean prediction
            weights = F.softmax(logits, dim=1)
            mu_w = (weights.unsqueeze(-1) * mu).sum(dim=1)
            
            # Compute RMSE per sample
            for i in range(xb.size(0)):
                yp = y_scaler.inverse_transform(mu_w[i].cpu().numpy().reshape(1, -1)).squeeze(0)
                yt = y_scaler.inverse_transform(yb[i].cpu().numpy().reshape(1, -1)).squeeze(0)
                all_rmse.append(trajectory_rmse(yp, yt))
    
    nll = total_nll / len(test_ds)
    mean_rmse = float(np.mean(all_rmse))
    median_rmse = float(np.median(all_rmse))
    
    return {
        "test_nll": nll,
        "test_rmse_mean": mean_rmse,
        "test_rmse_median": median_rmse,
        "n_samples": len(test_ds),
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Human Motion MDN")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-per-action", type=int, default=None,
                        help="Limit samples per action (None = all)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--sigma-penalty", type=float, default=DEFAULT_SIGMA_PENALTY)
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training; load saved model")
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HUMAN MOTION MDN")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    
    # ── Load data ──────────────────────────────────────────
    print("\n[LOAD] mini_amass dataset...")
    samples = load_mini_amass()
    print(f"  {len(samples)} samples loaded")
    
    # ── Prepare dataset ────────────────────────────────────
    print("\n[PREP] Building feature vectors...")
    t0 = time.time()
    X_train, Y_train, X_val, Y_val = prepare_dataset(
        samples, max_per_action=args.max_per_action)
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val: {Y_val.shape}")
    print(f"  Prep time: {time.time()-t0:.1f}s")
    
    # ── Scale ──────────────────────────────────────────────
    x_scaler = StandardScalerNP()
    y_scaler = StandardScalerNP()
    x_scaler.fit(X_train)
    y_scaler.fit(Y_train)
    
    # ── Build model ────────────────────────────────────────
    input_dim = X_train.shape[1]
    target_dim = Y_train.shape[1]
    
    model = JointPathMDN(
        input_dim=input_dim,
        target_dim=target_dim,
        n_components=args.n_components,
        hidden_dim=args.hidden_dim,
    )
    
    # ── Train ──────────────────────────────────────────────
    if not args.no_train:
        history = train_mdn(
            model=model,
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val,
            x_scaler=x_scaler, y_scaler=y_scaler,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            out_dir=out_dir,
        )
    else:
        ckpt = torch.load(out_dir / "human_motion_mdn.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        x_scaler.mean = ckpt["x_scaler_mean"]
        x_scaler.std = ckpt["x_scaler_std"]
        y_scaler.mean = ckpt["y_scaler_mean"]
        y_scaler.std = ckpt["y_scaler_std"]
        print(f"[LOAD] Model loaded from {out_dir / 'human_motion_mdn.pt'}")
    
    # ── Evaluate ───────────────────────────────────────────
    print("\n[EVAL] Evaluation on validation set...")
    metrics = evaluate_model(model, X_val, Y_val, x_scaler, y_scaler, device)
    print(f"  Test NLL:       {metrics['test_nll']:.4f}")
    print(f"  Test RMSE mean: {metrics['test_rmse_mean']:.6f}")
    print(f"  Test RMSE med:  {metrics['test_rmse_median']:.6f}")
    
    # Save metrics
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
