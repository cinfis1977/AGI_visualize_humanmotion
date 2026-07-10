#!/usr/bin/env python3
"""
Stage5A — Per-Joint MDN + Cross-Attention for Human Motion.

Architecture (faithful to baseball/goalkeeper Z2):
  1. Global base path per joint:  Q_j = mean(train_trajectories[:,:,j])
  2. Start-goal deformation:      D_{s,g}[Q_j]  (same formula as Z2)
  3. Per-joint MDN predicts RESIDUAL from deformed baseline
  4. Cross-attention coordinates joints at each timestep
  5. Trained autoregressively with teacher forcing

Synthetic walk: each joint follows a phase-shifted sine wave.
  - Pelvis: vertical bounce + lateral sway
  - Legs: pendulum swing (hip→knee→ankle chain)
  - Arms: opposite-phase pendulum
"""

from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

Array = np.ndarray


# ============================================================
# Deformation (same as Z2 goalkeeper)
# ============================================================

def deform_path_to_start_and_goal(path: Array, start: Array, goal: Array) -> Array:
    """D_{s,g}[Q] — baseball-style linear deformation."""
    Q = np.asarray(path, dtype=float)
    s = np.asarray(start, dtype=float).reshape(1, 3)
    g = np.asarray(goal, dtype=float).reshape(1, 3)
    alpha = np.linspace(0.0, 1.0, Q.shape[0], dtype=float).reshape(-1, 1)
    return Q - Q[0:1] + s + alpha * (g - (Q[-1:] - Q[0:1] + s))


def compute_bases(data: Array) -> Array:
    """Compute per-joint base paths: mean over training samples.
    data: (N, T, J*3) → returns (J, T, 3)
    """
    N, T, D = data.shape
    J = D // 3
    shaped = data.reshape(N, T, J, 3)
    mean_paths = np.nanmean(shaped, axis=0)  # (T, J, 3)
    return mean_paths.transpose(1, 0, 2)  # → (J, T, 3) for easy bases[j] access


# ============================================================
# Synthetic walk data generator
# ============================================================

def generate_synthetic_walk(n_samples: int = 100, n_frames: int = 32,
                            n_joints: int = 12, noise: float = 0.02,
                            seed: int = 42) -> Array:
    """
    Generate synthetic walking motion.
    Returns: (n_samples, n_frames, n_joints * 3) — flattened joint positions.

    Joint layout (12 joints, simplified from 24):
      0: pelvis    1: spine     2: neck
      3: L_hip     4: L_knee    5: L_ankle
      6: R_hip     7: R_knee    8: R_ankle
      9: L_shoulder 10: L_elbow 11: R_shoulder 12: R_elbow
    Actually use 12 for simplicity.
    """
    rng = np.random.default_rng(seed)
    dim = n_joints * 3

    # Base walk parameters (vary per sample)
    step_freqs = rng.uniform(0.8, 1.3, n_samples)  # steps per cycle
    stride_lengths = rng.uniform(0.6, 1.4, n_samples)
    step_heights = rng.uniform(0.05, 0.2, n_samples)  # foot lift height
    arm_swings = rng.uniform(0.1, 0.4, n_samples)
    pelvis_bounce = rng.uniform(0.02, 0.06, n_samples)
    lateral_sway = rng.uniform(0.02, 0.08, n_samples)

    data = np.zeros((n_samples, n_frames, dim), dtype=np.float32)

    for s in range(n_samples):
        t = np.linspace(0, 2 * np.pi * step_freqs[s], n_frames)
        stride = stride_lengths[s]

        # Pelvis (joint 0): vertical bounce + lateral sway + forward translation
        x0 = np.linspace(0, stride, n_frames)  # forward walk
        y0 = lateral_sway[s] * np.sin(t)        # left-right sway
        z0 = pelvis_bounce[s] * np.abs(np.sin(2 * t))  # double bounce per cycle

        # Legs: pendulum motion (hips→knees→ankles)
        # Phase offsets: L_hip at 0, R_hip at π (opposite)
        leg_amp = 0.4 * stride_lengths[s]
        knee_amp = 0.5 * stride_lengths[s]
        foot_lift = step_heights[s]

        # Left leg (joints 3,4,5)
        l_hip_x = -leg_amp * np.cos(t)
        l_hip_z = -0.3 + foot_lift * np.maximum(0, -np.cos(t))  # lift during swing
        l_knee_x = -knee_amp * np.cos(t)
        l_knee_z = -0.6 + foot_lift * 0.5 * np.maximum(0, -np.cos(t))
        l_ankle_x = 0.0
        l_ankle_z = -0.9 + foot_lift * 0.3 * np.maximum(0, np.cos(t))

        # Right leg (joints 6,7,8) — opposite phase
        r_hip_x = leg_amp * np.cos(t)
        r_hip_z = -0.3 + foot_lift * np.maximum(0, np.cos(t))
        r_knee_x = knee_amp * np.cos(t)
        r_knee_z = -0.6 + foot_lift * 0.5 * np.maximum(0, np.cos(t))
        r_ankle_x = 0.0
        r_ankle_z = -0.9 + foot_lift * 0.3 * np.maximum(0, -np.cos(t))

        # Arms (joints 9,10,11,12) — opposite phase to legs
        arm_amp = arm_swings[s]
        l_shoulder_x = -arm_amp * np.cos(t + np.pi)  # opposite to left leg
        l_elbow_x = -arm_amp * 0.7 * np.cos(t + np.pi)
        r_shoulder_x = arm_amp * np.cos(t)  # opposite to right leg
        r_elbow_x = arm_amp * 0.7 * np.cos(t)

        # Spine and neck: minor oscillation
        spine_x = 0.02 * np.sin(t)
        neck_x = 0.03 * np.sin(t)

        # Assemble: [x, y, z] per joint
        joints = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
        # 0: pelvis
        joints[:, 0, 0] = x0;  joints[:, 0, 1] = y0;  joints[:, 0, 2] = z0 + 0.9
        # 1: spine
        joints[:, 1, 0] = x0 + spine_x; joints[:, 1, 1] = y0; joints[:, 1, 2] = z0 + 1.2
        # 2: neck
        joints[:, 2, 0] = x0 + neck_x; joints[:, 2, 1] = y0; joints[:, 2, 2] = z0 + 1.5
        # 3: L_hip
        joints[:, 3, 0] = x0 + l_hip_x; joints[:, 3, 1] = y0 - 0.1; joints[:, 3, 2] = z0 + l_hip_z
        # 4: L_knee
        joints[:, 4, 0] = x0 + l_knee_x; joints[:, 4, 1] = y0 - 0.1; joints[:, 4, 2] = z0 + l_knee_z
        # 5: L_ankle
        joints[:, 5, 0] = x0 + l_ankle_x; joints[:, 5, 1] = y0 - 0.1; joints[:, 5, 2] = z0 + l_ankle_z
        # 6: R_hip
        joints[:, 6, 0] = x0 + r_hip_x; joints[:, 6, 1] = y0 + 0.1; joints[:, 6, 2] = z0 + r_hip_z
        # 7: R_knee
        joints[:, 7, 0] = x0 + r_knee_x; joints[:, 7, 1] = y0 + 0.1; joints[:, 7, 2] = z0 + r_knee_z
        # 8: R_ankle
        joints[:, 8, 0] = x0 + r_ankle_x; joints[:, 8, 1] = y0 + 0.1; joints[:, 8, 2] = z0 + r_ankle_z
        # 9: L_shoulder
        joints[:, 9, 0] = x0 + l_shoulder_x; joints[:, 9, 1] = y0 - 0.2; joints[:, 9, 2] = z0 + 1.4
        # 10: L_elbow
        joints[:, 10, 0] = x0 + l_elbow_x; joints[:, 10, 1] = y0 - 0.2; joints[:, 10, 2] = z0 + 1.1
        # 11: R_shoulder
        joints[:, 11, 0] = x0 + r_shoulder_x; joints[:, 11, 1] = y0 + 0.2; joints[:, 11, 2] = z0 + 1.4
        # 12: R_elbow  (we said 12 joints, but let's use 12: 0-11)
        # Actually we have 13 listed. Let's drop neck (make it 12 exactly)
        # 0:pelvis 1:spine 2:L_hip 3:L_knee 4:L_ankle 5:R_hip 6:R_knee 7:R_ankle
        # 8:L_shoulder 9:L_elbow 10:R_shoulder 11:R_elbow
        # That's 12. Let me renumber.

        data[s] = joints.reshape(n_frames, -1)[:, :dim]  # trim to n_joints*3

    return data + rng.normal(0, noise, data.shape).astype(np.float32)


def generate_synthetic_walk_12j(n_samples=100, n_frames=32, noise=0.02, seed=42):
    """12-joint synthetic walk with proper indexing."""
    rng = np.random.default_rng(seed)
    n_joints = 12
    dim = n_joints * 3  # 36

    step_freqs = rng.uniform(0.8, 1.3, n_samples)
    stride_lengths = rng.uniform(0.6, 1.4, n_samples)
    foot_lifts = rng.uniform(0.05, 0.2, n_samples)
    arm_swings = rng.uniform(0.1, 0.4, n_samples)
    bounce = rng.uniform(0.02, 0.06, n_samples)
    sway = rng.uniform(0.02, 0.08, n_samples)

    data = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)

    for s in range(n_samples):
        t = np.linspace(0, 2 * np.pi * step_freqs[s], n_frames)
        stride = stride_lengths[s]
        x_fwd = np.linspace(0, stride, n_frames)
        y_sway = sway[s] * np.sin(t)
        z_bounce = bounce[s] * np.abs(np.sin(2 * t))

        # 0: pelvis
        data[s, :, 0] = np.column_stack([x_fwd, y_sway, z_bounce + 0.9])
        # 1: spine (follows pelvis)
        data[s, :, 1] = np.column_stack([x_fwd + 0.01*np.sin(t), y_sway, z_bounce + 1.2])

        leg_amp = 0.4 * stride
        knee_amp = 0.5 * stride
        # 2: L_hip, 3: L_knee, 4: L_ankle  (left leg, phase=0)
        l_swing = foot_lifts[s] * np.maximum(0, -np.cos(t))
        data[s, :, 2] = np.column_stack([x_fwd - leg_amp*np.cos(t), y_sway - 0.1, 0.6 + l_swing])
        data[s, :, 3] = np.column_stack([x_fwd - knee_amp*np.cos(t), y_sway - 0.1, 0.3 + 0.5*l_swing])
        data[s, :, 4] = np.column_stack([x_fwd, y_sway - 0.1, 0.0 + 0.3*l_swing])

        # 5: R_hip, 6: R_knee, 7: R_ankle  (right leg, phase=pi)
        r_swing = foot_lifts[s] * np.maximum(0, np.cos(t))
        data[s, :, 5] = np.column_stack([x_fwd + leg_amp*np.cos(t), y_sway + 0.1, 0.6 + r_swing])
        data[s, :, 6] = np.column_stack([x_fwd + knee_amp*np.cos(t), y_sway + 0.1, 0.3 + 0.5*r_swing])
        data[s, :, 7] = np.column_stack([x_fwd, y_sway + 0.1, 0.0 + 0.3*r_swing])

        # 8: L_shoulder, 9: L_elbow (left arm, opposite to left leg)
        a_amp = arm_swings[s]
        data[s, :, 8] = np.column_stack([x_fwd - a_amp*np.cos(t+np.pi), y_sway-0.2, np.full(n_frames, 1.35)])
        data[s, :, 9] = np.column_stack([x_fwd - 0.7*a_amp*np.cos(t+np.pi), y_sway-0.2, np.full(n_frames, 1.05)])

        # 10: R_shoulder, 11: R_elbow (right arm, opposite to right leg)
        data[s, :, 10] = np.column_stack([x_fwd + a_amp*np.cos(t), y_sway+0.2, np.full(n_frames, 1.35)])
        data[s, :, 11] = np.column_stack([x_fwd + 0.7*a_amp*np.cos(t), y_sway+0.2, np.full(n_frames, 1.05)])

    data += rng.normal(0, noise, data.shape).astype(np.float32)
    return data.reshape(n_samples, n_frames, -1), n_joints


# ============================================================
# Per-Joint MDN
# ============================================================

class JointMDN(nn.Module):
    """Small MDN for one joint. Predicts Δpos (3D) from joint state + context."""
    def __init__(self, state_dim: int, context_dim: int, n_components: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        in_dim = state_dim + context_dim  # joint's own state + cross-attention context
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.logits = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * 3)  # Δx, Δy, Δz
        self.log_sigma = nn.Linear(hidden_dim, n_components)

        # Conservative init (baseball style)
        nn.init.zeros_(self.mu.bias)
        nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.log_sigma.bias)

    def forward(self, state_j: torch.Tensor, context_j: torch.Tensor):
        """
        state_j:   (B, state_dim)  — joint's own state at time t
        context_j: (B, context_dim) — cross-attention output for this joint
        Returns: logits (B,K), mu (B,K,3), log_sigma (B,K)
        """
        h = self.net(torch.cat([state_j, context_j], dim=-1))
        K = self.logits.out_features
        logits = self.logits(h)
        mu = self.mu(h).view(-1, K, 3)
        log_sigma = self.log_sigma(h).clamp(-4.0, 3.0)
        return logits, mu, log_sigma


class CrossAttentionCoordinator(nn.Module):
    """Multi-head cross-attention: each joint attends to all other joints."""
    def __init__(self, joint_state_dim: int, n_heads: int = 4, n_joints: int = 12):
        super().__init__()
        self.n_joints = n_joints
        self.attn = nn.MultiheadAttention(
            embed_dim=joint_state_dim,
            num_heads=n_heads,
            batch_first=True,  # (B, J, D)
        )

    def forward(self, joint_states: torch.Tensor):
        """
        joint_states: (B, n_joints, state_dim)
        Returns: (B, n_joints, state_dim) — context-aware states
        """
        context, _ = self.attn(joint_states, joint_states, joint_states)
        return context  # residual connection handled outside


class PerJointMDNWithCoordination(nn.Module):
    """
    Full model: J joint MDNs + 1 cross-attention coordinator.

    Faithful to baseball/goalkeeper Z2:
      baseline_j = D_{s,g}[base_j]  (deformed per-joint base)
      MDN_j predicts RESIDUAL:  true_pos = baseline + residual

    At each timestep:
      1. Each joint encodes its state + baseline position
      2. Cross-attention: all joints share information
      3. Each joint's MDN predicts residual (deviation from baseline)
      4. Next position = baseline + residual
    """
    def __init__(self, n_joints: int = 12, n_components: int = 4,
                 joint_state_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.n_joints = n_joints
        self.joint_state_dim = joint_state_dim

        # Per-joint MDNs (predict residual, not absolute)
        self.mdns = nn.ModuleList([
            JointMDN(state_dim=joint_state_dim, context_dim=joint_state_dim,
                     n_components=n_components)
            for _ in range(n_joints)
        ])

        # Cross-attention coordinator
        self.coordinator = CrossAttentionCoordinator(
            joint_state_dim=joint_state_dim, n_heads=n_heads, n_joints=n_joints
        )

        # Joint state encoder: [pos(3), base_pos(3), vel(3), goal(3), action(3)] → state_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(15, joint_state_dim), nn.GELU(),
            nn.Linear(joint_state_dim, joint_state_dim),
        )

    def encode_state(self, pos: torch.Tensor, baseline_pos: torch.Tensor,
                     vel: torch.Tensor, goal: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """Encode per-joint state including baseline position.
        pos:      (B, J, 3)
        baseline: (B, J, 3)
        vel:      (B, J, 3)
        goal:     (B, J, 3)
        action:   (B, 3) broadcast to (B, J, 3)
        Returns:  (B, J, joint_state_dim)
        """
        B, J = pos.shape[:2]
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        feat = torch.cat([pos, baseline_pos, vel, goal, action_exp], dim=-1)  # (B, J, 15)
        return self.state_encoder(feat)

    def forward_step(self, pos: torch.Tensor, baseline_pos: torch.Tensor,
                     vel: torch.Tensor, goal: torch.Tensor, action: torch.Tensor):
        """
        Single timestep: predict RESIDUAL from baseline for all joints.
        Returns: logits (B,J,K), mu (B,J,K,3), log_sigma (B,J,K)
        """
        B, J = pos.shape[:2]
        states = self.encode_state(pos, baseline_pos, vel, goal, action)
        context = self.coordinator(states)

        all_logits, all_mu, all_sigma = [], [], []
        for j in range(J):
            logits_j, mu_j, sigma_j = self.mdns[j](states[:, j, :], context[:, j, :])
            all_logits.append(logits_j.unsqueeze(1))
            all_mu.append(mu_j.unsqueeze(1))
            all_sigma.append(sigma_j.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)
        mu = torch.cat(all_mu, dim=1)
        log_sigma = torch.cat(all_sigma, dim=1)
        return logits, mu, log_sigma


# ============================================================
# Training
# ============================================================

def mdn_nll_joint(logits, mu, log_sigma, target_delta):
    """Negative log-likelihood for per-joint MDN.
    logits:     (B, J, K)
    mu:         (B, J, K, 3)
    log_sigma:  (B, J, K)
    target_delta: (B, J, 3) — ground truth Δpos
    """
    B, J, K = logits.shape
    diff = target_delta.unsqueeze(2) - mu  # (B, J, K, 3)
    inv_var = torch.exp(-2.0 * log_sigma).unsqueeze(-1)  # (B, J, K, 1)
    log_prob = -0.5 * (diff * diff * inv_var).sum(dim=-1) - 3.0 * log_sigma - 0.5 * 3.0 * math.log(2 * math.pi)
    # Sum over joints (independent), then mixture
    log_prob_joint = log_prob.sum(dim=1)  # (B, K) — sum log-prob across joints
    log_pi = F.log_softmax(logits.mean(dim=1), dim=-1)  # (B, K) — shared mixing weights (averaged)
    # Actually, let's use per-sample mixture weights (mean across joints)
    log_pi_avg = F.log_softmax(logits.mean(dim=1), dim=-1)  # (B, K)
    nll = -torch.logsumexp(log_pi_avg + log_prob_joint, dim=-1).mean()
    return nll


def train_model(model, data, bases, n_epochs=200, batch_size=32, lr=1e-3, device="cpu"):
    """Train with base-deform residual targets (like Z2)."""
    n_samples, n_frames, dim = data.shape
    n_joints = model.n_joints

    X_data = torch.tensor(data, dtype=torch.float32, device=device)
    X_data = X_data.view(n_samples, n_frames, n_joints, 3)

    # Compute deformed baselines for all samples: D_{s,g}[base_j] for each joint
    # bases shape: (T, J, 3) from compute_bases
    baselines = np.zeros((n_samples, n_frames, n_joints, 3), dtype=np.float32)
    for s in range(n_samples):
        for j in range(n_joints):
            start_j = X_data[s, 0, j].cpu().numpy()
            goal_j = X_data[s, -1, j].cpu().numpy()
            base_j = bases[j]  # (T, 3) for joint j
            deformed = deform_path_to_start_and_goal(base_j, start_j, goal_j)
            baselines[s, :, j] = deformed
    baselines = torch.tensor(baselines, dtype=torch.float32, device=device)

    # Velocities of the baseline (for input encoding)
    base_vel = torch.zeros_like(baselines)
    base_vel[:, 1:] = baselines[:, 1:] - baselines[:, :-1]

    # Velocities of true data
    vel = torch.zeros_like(X_data)
    vel[:, 1:] = X_data[:, 1:] - X_data[:, :-1]

    # Goals = last frame
    goals = X_data[:, -1, :, :].unsqueeze(1).expand(-1, n_frames, -1, -1)

    # Action encoding
    actions = torch.zeros(n_samples, n_frames, 3, device=device)
    actions[:, :, 0] = 1.0  # "walk forward"

    # Targets: RESIDUAL from baseline (not absolute Δpos!)
    # residual = true_next_pos - baseline_next_pos
    # But we train per-step: at time t, we predict residual = true_pos[t+1] - baseline[t+1]
    targets = X_data - baselines  # (N, T, J, 3) — full residual

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0

        for st in range(0, n_samples, batch_size):
            b = perm[st:st + batch_size]
            B = len(b)

            t0 = torch.randint(0, n_frames - 2, (1,)).item()
            t_end = min(t0 + 8, n_frames - 1)

            loss_t = 0.0
            for t in range(t0, t_end):
                pos_t = X_data[b, t]
                base_t = baselines[b, t]
                base_v_t = base_vel[b, t]
                goal_t = goals[b, t]
                act_t = actions[b, t]
                target_t = targets[b, t + 1]  # residual to predict

                logits, mu, log_sigma = model.forward_step(pos_t, base_t, base_v_t, goal_t, act_t)
                loss_t += mdn_nll_joint(logits, mu, log_sigma, target_t)

            loss = loss_t / (t_end - t0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * B

        avg_loss = total_loss / n_samples
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}: loss={avg_loss:.4f}  best={best_loss:.4f}")

    return best_loss, baselines


def generate_motion(model, bases, start_pose, goal_pose, n_frames=32,
                    action_vec=None, device="cpu"):
    """Autoregressive generation: from start to goal, using base-deform + residual."""
    model.eval()
    n_joints = model.n_joints

    pos = torch.tensor(start_pose, dtype=torch.float32, device=device).unsqueeze(0)  # (1, J, 3)
    goal = torch.tensor(goal_pose, dtype=torch.float32, device=device).unsqueeze(0)
    if action_vec is None:
        action = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)
    else:
        action = torch.tensor(action_vec, dtype=torch.float32, device=device).unsqueeze(0)

    # Pre-compute deformed baselines for this sample
    bases_np = bases  # (J, T, 3)
    baselines = np.zeros((n_joints, n_frames, 3), dtype=np.float32)
    for j in range(n_joints):
        s_j = start_pose[j]
        g_j = goal_pose[j]
        b_j = bases_np[j]
        baselines[j] = deform_path_to_start_and_goal(b_j, s_j, g_j)
    baselines_t = torch.tensor(baselines, dtype=torch.float32, device=device).unsqueeze(0)  # (1, J, T, 3)
    base_vel = torch.zeros_like(baselines_t)
    base_vel[:, :, 1:] = baselines_t[:, :, 1:] - baselines_t[:, :, :-1]

    trajectory = [pos.detach().cpu().numpy()]
    vel = torch.zeros(1, n_joints, 3, device=device)

    for t in range(n_frames - 1):
        base_pos_t = baselines_t[:, :, t]       # (1, J, 3)
        base_vel_t = base_vel[:, :, t]

        logits, mu, log_sigma = model.forward_step(pos, base_pos_t, base_vel_t, goal, action)
        probs = F.softmax(logits, dim=-1)
        residual = (probs.unsqueeze(-1) * mu).sum(dim=2)  # (1, J, 3)

        # Next position = baseline + residual
        next_base = baselines_t[:, :, t + 1]
        pos = next_base + residual.detach()
        vel = residual.detach()
        trajectory.append(pos.detach().cpu().numpy())

    return np.concatenate(trajectory, axis=0)  # (T, J, 3)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-frames", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if torch is None:
        print("PyTorch not available.")
        return

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Generating synthetic walk data...")
    data_flat, n_joints = generate_synthetic_walk_12j(
        n_samples=args.n_samples, n_frames=args.n_frames, noise=0.02, seed=args.seed
    )
    print(f"Data: {data_flat.shape}  ({n_joints} joints × 3D = {n_joints*3})")

    # Train/val split
    n_train = int(0.8 * args.n_samples)
    train_data = data_flat[:n_train]
    val_data = data_flat[n_train:]

    # Compute per-joint base paths from TRAINING data only
    bases = compute_bases(train_data)  # (J, T, 3)
    print(f"Bases: {bases.shape}")

    print(f"\nBuilding model: {n_joints} joints, each with 4-component MDN + cross-attention...")
    print("  Architecture: base -> D_sg[base] -> MDN(residual)  [faithful to Z2]")
    model = PerJointMDNWithCoordination(
        n_joints=n_joints, n_components=4,
        joint_state_dim=32, n_heads=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\nTraining on {n_train} samples...")
    t0 = time.time()
    best_loss, baselines = train_model(model, train_data, bases, n_epochs=args.n_epochs,
                                       batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"Training complete: {time.time()-t0:.1f}s, best_loss={best_loss:.4f}")

    # Test generation
    print("\nGenerating test motion...")
    test_sample = val_data[0].reshape(args.n_frames, n_joints, 3)
    start = test_sample[0]
    goal = test_sample[-1]

    generated = generate_motion(model, bases, start, goal, n_frames=args.n_frames, device=device)

    # Compare
    true_rmse = np.sqrt(np.mean((test_sample - generated) ** 2))
    goal_err = np.sqrt(np.mean((generated[-1] - goal) ** 2))
    print(f"  True vs generated RMSE: {true_rmse:.4f}")
    print(f"  Goal error: {goal_err:.4f}")

    # Per-joint RMSE
    print("\nPer-joint RMSE:")
    for j in range(n_joints):
        j_rmse = np.sqrt(np.mean((test_sample[:, j, :] - generated[:, j, :]) ** 2))
        print(f"  joint {j:2d}: {j_rmse:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
