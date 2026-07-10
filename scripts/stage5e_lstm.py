#!/usr/bin/env python3
"""
Stage5E-LSTM — LSTM temporal encoder + per-timestep MDN.
LSTM provides phase awareness (stance/swing), MDN predicts per frame.
Keeps numpy FK loss (non-differentiable) for CPU compatibility.
"""

from __future__ import annotations
import argparse, math, sys, time, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch, torch.nn as nn, torch.nn.functional as F
from smpl_fk import fk_smpl_full

Array = np.ndarray

# ════════════════════════════════════════════════════════════
# Data loader — same as stage5e
# ════════════════════════════════════════════════════════════

def load_data(data_dir, n_files=200, n_frames=96, min_translation=1.0, seed=42):
    base = Path(data_dir)
    all_files = [f for f in base.rglob("*.npz") if "walk" in f.name.lower()]
    rng = np.random.default_rng(seed)
    sequences = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            pb = d["pose_body"]; tr = d["trans"]
            if pb.shape[0] < n_frames: continue
            if tr[:,0].max()-tr[:,0].min() < min_translation: continue
            idx = np.linspace(0, pb.shape[0]-1, n_frames, dtype=int)
            full_rot = np.concatenate([d["root_orient"][idx], pb[idx],
                                       np.zeros((n_frames,6), dtype=np.float32)], axis=1)
            sequences.append(np.concatenate([full_rot, tr[idx]], axis=1))
        except: continue
    if n_files > 0 and len(sequences) > n_files:
        sequences = list(rng.choice(sequences, n_files, replace=False))
    data = np.stack(sequences, axis=0).astype(np.float32)
    print(f"Loaded {len(sequences)} walks (24 joints)")
    return data, 24


def compute_mean_base(data_rot, n_base=10):
    """Per-joint mean base: (J, T, 3)."""
    N, T, D = data_rot.shape; J = D // 3
    pool = data_rot[:min(n_base,N)].reshape(-1, T, J, 3)
    mean = np.nanmean(pool, axis=0)
    return mean.transpose(1,0,2).astype(np.float32)


def baseball_deform(path, start, goal, gamma=1.0):
    Q = np.asarray(path, dtype=np.float32); T = Q.shape[0]
    s = np.asarray(start, dtype=np.float32).reshape(1,-1)
    g = np.asarray(goal, dtype=np.float32).reshape(1,-1)
    a = np.linspace(0,1,T,dtype=np.float32).reshape(-1,1)
    return Q - Q[0:1] + s + gamma*a*(g - (Q[-1:]-Q[0:1]+s))


# ════════════════════════════════════════════════════════════
# Model — same architecture as stage5e
# ════════════════════════════════════════════════════════════

class PerTimestepMDN(nn.Module):
    """Predicts K Gaussian components per timestep (not all T at once)."""
    def __init__(self, in_dim, K=4, hidden_dim=64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, K*3)
        nn.init.normal_(self.mu_head.weight, 0, 0.01)
        nn.init.zeros_(self.mu_head.bias)

    def forward(self, x):
        """x: (..., in_dim) → mu: (..., K, 3)"""
        h = self.net(x)
        return self.mu_head(h).reshape(*x.shape[:-1], self.K, 3)


class BaseballVertexMDN(nn.Module):
    def __init__(self, n_joints=24, n_timesteps=96, n_components=4, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints; self.n_timesteps = n_timesteps; self.K = n_components
        self.joint_encoder = nn.Sequential(nn.Linear(15, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        # Bidirectional LSTM over time (processes each joint independently)
        self.lstm = nn.LSTM(feature_dim*2 + 4, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # Per-joint per-timestep MDN heads
        lstm_out = hidden_dim * 2  # bidirectional
        self.mdn_heads = nn.ModuleList([
            PerTimestepMDN(lstm_out, n_components, hidden_dim=64)
            for _ in range(n_joints)
        ])
        # Logits (component weights) — shared across time, per joint
        self.logits_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(lstm_out, 64), nn.GELU(), nn.Linear(64, n_components))
            for _ in range(n_joints)
        ])

    def forward(self, start_rot, goal_rot, action, base_features, apply_smooth=False):
        B, J = start_rot.shape[:2]; K = self.K; T = self.n_timesteps
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        # Per-joint static features: (B, J, 15)
        feat_static = torch.cat([start_rot, goal_rot, base_features, action_exp], dim=-1)
        encoded = self.joint_encoder(feat_static)  # (B, J, feature_dim)

        # Cross-attention over joints (static)
        context, _ = self.attn(encoded, encoded, encoded)  # (B, J, feature_dim)
        joint_feat = torch.cat([encoded, context], dim=-1)  # (B, J, 2*feature_dim)

        # Expand to time: repeat static features for each timestep
        D = joint_feat.shape[-1]  # 2*feature_dim
        joint_feat_t = joint_feat.unsqueeze(2).expand(B, J, T, D)
        # LSTM over time: process each joint independently
        # Temporal phase encoding: sinusoidal position signal so LSTM
        # sees DIFFERENT input at each timestep (critical fix).
        # Without this, LSTM gets identical static input T times → useless.
        t_norm = torch.linspace(0, 1, T, device=joint_feat.device, dtype=joint_feat.dtype)
        phase = torch.stack([torch.sin(2*np.pi*t_norm), torch.cos(2*np.pi*t_norm),
                            torch.sin(4*np.pi*t_norm), torch.cos(4*np.pi*t_norm)], dim=-1)  # (T, 4)
        phase = phase.unsqueeze(0).unsqueeze(1).expand(B, J, T, 4)  # (B, J, T, 4)
        # Concatenate phase to static features per timestep
        joint_feat_t = torch.cat([joint_feat_t, phase], dim=-1)  # (B, J, T, D+4)
        Dp = joint_feat_t.shape[-1]
        x_lstm = joint_feat_t.reshape(B*J, T, Dp)
        out_lstm, _ = self.lstm(x_lstm)  # (B*J, T, 2*H)
        out_lstm = out_lstm.reshape(B, J, T, -1)  # (B, J, T, 2*H)

        # Per-joint MDN: predict mu per timestep
        all_mu = []
        for j in range(J):
            mu_j = self.mdn_heads[j](out_lstm[:, j, :, :])  # (B, T, K, 3)
            all_mu.append(mu_j.unsqueeze(1))  # (B, 1, T, K, 3)
        mu = torch.cat(all_mu, dim=1)  # (B, J, T, K, 3)

        # Logits: time-pooled → one set of component weights per joint
        time_pooled = out_lstm.mean(dim=2)  # (B, J, 2*H)
        all_logits = []
        for j in range(J):
            lj = self.logits_heads[j](time_pooled[:, j, :])  # (B, K)
            all_logits.append(lj.unsqueeze(1))
        logits = torch.cat(all_logits, dim=1)  # (B, J, K)

        # Reshape mu to match old format for training compatibility: (B, J, K, T, 3)
        mu = mu.permute(0, 1, 3, 2, 4)  # (B, J, K, T, 3)
        return logits, mu


# ════════════════════════════════════════════════════════════
# Training — FIX #1: no-scale raw radian residuals
# ════════════════════════════════════════════════════════════

def train_model(model, data75, bases, epochs=200, bs=16, lr=1e-3, device="cpu",
                aux_weights=None, winner_realism=None, fk_loss_weight=5.0):
    N, T, D = data75.shape; J = model.n_joints
    rot_data = data75[:,:,:72].reshape(N, T, J, 3)
    trans_data = data75[:,:,72:75]
    R = torch.tensor(rot_data, dtype=torch.float32, device=device)
    Tr = torch.tensor(trans_data, dtype=torch.float32, device=device)

    # Pre-compute deformed baselines
    print("  Computing baselines...")
    bl = np.zeros((N, T, J, 3), dtype=np.float32)
    for n in range(N):
        for j in range(J):
            bl[n,:,j] = baseball_deform(bases[j], rot_data[n,0,j], rot_data[n,-1,j])
    BL = torch.tensor(bl, dtype=torch.float32, device=device)

    # FIX #1: NO-SCALE residuals — raw radians, no standardization
    res = R.permute(0,2,1,3) - BL.permute(0,2,1,3)  # (N,J,T,3)
    Y = res.reshape(N, J, -1)  # raw radian residuals

    # Input: start+goal + base features (scaled for conditioning only)
    sg = np.concatenate([rot_data[:,0], rot_data[:,-1]], -1).reshape(N, J, 6)
    sg_mean = sg.reshape(-1,6).mean(0).astype(np.float32)
    sg_std = sg.reshape(-1,6).std(0).astype(np.float32) + 1e-8

    x_scaler = type('obj', (object,), {
        'mean': sg_mean, 'std': sg_std,
        'transform': lambda self, X: (X - self.mean) / self.std,
        'inverse': lambda self, X: X * self.std + self.mean,
    })()

    sg_s = x_scaler.transform(sg.reshape(-1,6)).reshape(N, J, 6)
    Ss = torch.tensor(sg_s[...,:3], dtype=torch.float32, device=device)
    Gs = torch.tensor(sg_s[...,3:], dtype=torch.float32, device=device)

    # Base features
    base_feat = np.zeros((N, J, 6), dtype=np.float32)
    for j in range(J):
        bj = bases[j]
        base_feat[:, j, 0] = np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum()
        base_feat[:, j, 1] = np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if T>2 else 0
        base_feat[:, j, 3] = bj[:,2].max()
        base_feat[:, j, 4] = bj[:,2].std()
        base_feat[:, j, 5] = bj[-1,2] - bj[0,2]
    Bf = torch.tensor(base_feat, dtype=torch.float32, device=device)

    actions = torch.zeros(N, 3, device=device); actions[:,0] = 1.0

    nv = max(1, int(0.15*N))
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    p = torch.randperm(N, generator=g, device="cpu")
    vi, ti = p[:nv], p[nv:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf"); best_state = None
    K = 4

    print(f"  Stage5E-LSTM: temporal encoder + per-timestep MDN | train={len(ti)} val={nv}")

    for ep in range(1, epochs+1):
        model.train()
        tp = torch.randperm(len(ti), device="cpu")
        total, cnt = 0.0, 0
        for st in range(0, len(ti), bs):
            b = ti[tp[st:st+bs]]
            logits, mu = model(Ss[b], Gs[b], actions[b], Bf[b])
            y = Y[b].view(-1, J, T, 3).unsqueeze(2)  # (B,J,1,T,3)

            diff = y - mu  # (B,J,K,T,3)
            err = (diff**2).mean((-2,-1))  # (B,J,K)
            win = err.argmin(-1)
            w = torch.full_like(err, 0.05/(K-1))
            w.scatter_(-1, win.unsqueeze(-1), 0.95)
            recon = (w*err).sum(-1).mean()
            gate = sum(F.cross_entropy(logits[:,j], win[:,j]) for j in range(J))/J

            # FK 3D position loss — ALL batch samples (not just batch[0])
            B_cur = len(b)
            mu_win = mu.gather(2, win.view(B_cur, J, 1, 1, 1).expand(B_cur, J, 1, T, 3)).squeeze(2)  # (B,J,T,3)
            pred_rot_all = BL[b] + mu_win.permute(0, 2, 1, 3)  # (B,T,J,3)
            true_rot_all = R[b]  # (B,T,J,3)
            fk_errors = []
            for bi in range(B_cur):
                pr = pred_rot_all[bi].reshape(T, 72).cpu().numpy().astype(np.float32)
                tr_np = true_rot_all[bi].reshape(T, 72).cpu().numpy().astype(np.float32)
                p3d = fk_smpl_full(pr, trans_data[b[bi]].astype(np.float32))
                t3d = fk_smpl_full(tr_np, trans_data[b[bi]].astype(np.float32))
                fk_errors.append(np.mean((p3d - t3d)**2))
            fk_loss = torch.tensor(np.mean(fk_errors), dtype=torch.float32, device=device)

            # Amplitude matching — ALL batch samples
            pred_rot_np = pred_rot_all.cpu().numpy()
            true_rot_np = true_rot_all.cpu().numpy()
            pred_amp = np.std(pred_rot_np, axis=1).mean()
            true_amp = np.std(true_rot_np, axis=1).mean()
            amp_loss = torch.tensor((pred_amp - true_amp)**2, dtype=torch.float32, device=device)

            # Velocity matching — ALL batch samples
            pv = pred_rot_np[:, 1:, :, :] - pred_rot_np[:, :-1, :, :]
            tv = true_rot_np[:, 1:, :, :] - true_rot_np[:, :-1, :, :]
            vel_loss = torch.tensor(np.mean((pv - tv)**2), dtype=torch.float32, device=device)

            # Acceleration matching — ALL batch samples
            pa = pv[:, 1:, :, :] - pv[:, :-1, :, :]
            ta = tv[:, 1:, :, :] - tv[:, :-1, :, :]
            acc_loss = torch.tensor(np.mean((pa - ta)**2), dtype=torch.float32, device=device)

            loss = 100.0*recon + 0.1*gate + fk_loss_weight*fk_loss \
                 + 30.0*amp_loss + 5.0*vel_loss + 2.0*acc_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()*len(b); cnt += len(b)

        # Validation metrics — compute properly on val set
        model.eval()
        with torch.no_grad():
            vl, vm = model(Ss[vi], Gs[vi], actions[vi], Bf[vi])
            vd = Y[vi].view(-1,J,T,3).unsqueeze(2) - vm
            ve = (vd**2).mean((-2,-1))
            vw = ve.argmin(-1)
            vw2 = torch.full_like(ve, 0.05/(K-1))
            vw2.scatter_(-1, vw.unsqueeze(-1), 0.95)
            val_loss = (vw2*ve).sum(-1).mean().item()

            # FK + amp/vel/acc on VALIDATION set (not training batch!)
            mu_win_v = vm.gather(2, vw.view(len(vi), J, 1, 1, 1).expand(len(vi), J, 1, T, 3)).squeeze(2)
            pred_val = BL[vi] + mu_win_v.permute(0, 2, 1, 3)
            true_val = R[vi]
            pv_np = pred_val.cpu().numpy(); tv_np = true_val.cpu().numpy()
            val_amp = float(np.mean((np.std(pv_np, axis=1) - np.std(tv_np, axis=1))**2))
            pv_v = pv_np[:, 1:, :, :] - pv_np[:, :-1, :, :]
            tv_v = tv_np[:, 1:, :, :] - tv_np[:, :-1, :, :]
            val_vel = float(np.mean((pv_v - tv_v)**2))
            pa_v = pv_v[:, 1:, :, :] - pv_v[:, :-1, :, :]
            ta_v = tv_v[:, 1:, :, :] - tv_v[:, :-1, :, :]
            val_acc = float(np.mean((pa_v - ta_v)**2))

        combined_val = val_loss + 0.5*val_amp + 0.08*val_vel + 0.03*val_acc

        if combined_val < best_val - 1e-7:
            best_val = combined_val
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

        if ep==1 or ep%10==0 or ep==epochs:
            print(f"  ep {ep:4d}: train={total/max(cnt,1):.4f} val={val_loss:.6f} "
                  f"fk={float(fk_loss):.6f} amp={float(amp_loss):.6f} best={best_val:.6f}")

    if best_state: model.load_state_dict(best_state)
    return best_val, x_scaler, None  # y_scaler=None (no-scale)


# ════════════════════════════════════════════════════════════
# Generation — FIX #3: boundary-locked + safe-improvement
# ════════════════════════════════════════════════════════════

def generate_motion(model, bases, start, goal, trans, x_scaler=None, y_scaler=None, device="cpu",
                     residual_scale=1.0, noise_std=0.0):
    model.eval()
    J, T = 24, model.n_timesteps
    s = np.asarray(start, np.float32).reshape(J,3)
    g = np.asarray(goal, np.float32).reshape(J,3)

    if x_scaler is not None:
        sg = x_scaler.transform(np.concatenate([s,g],-1))
        S = torch.tensor(sg[:,:3], dtype=torch.float32, device=device).unsqueeze(0)
        G = torch.tensor(sg[:,3:], dtype=torch.float32, device=device).unsqueeze(0)
    else:
        S = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        G = torch.tensor(g, dtype=torch.float32, device=device).unsqueeze(0)

    action = torch.zeros(1,3,device=device); action[:,0]=1.0
    # Compute REAL base features (not zeros! model was trained with these)
    bf = torch.zeros(1, J, 6, device=device)
    for j in range(J):
        bj = bases[j]
        bf[0, j, 0] = torch.tensor(np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum(), dtype=torch.float32)
        bf[0, j, 1] = torch.tensor(np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if T>2 else 0.0, dtype=torch.float32)
        bf[0, j, 3] = torch.tensor(bj[:,2].max(), dtype=torch.float32)
        bf[0, j, 4] = torch.tensor(bj[:,2].std(), dtype=torch.float32)
        bf[0, j, 5] = torch.tensor(bj[-1,2] - bj[0,2], dtype=torch.float32)

    with torch.no_grad():
        logits, mu = model(S, G, action, bf)
        k = logits[0].argmax(-1)
        mu_k = torch.stack([mu[0,j,k[j]] for j in range(J)])  # (J,T,3)

    res = mu_k.cpu().numpy()  # raw radian residuals (no inverse scaler needed)

    # HACK: scale residuals to compensate for Gen/True < 1.0
    if residual_scale != 1.0:
        res = res * residual_scale

    # Add micro-variation noise (anti-stiffness: simulates high-frequency natural jitter)
    if noise_std > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_std, res.shape).astype(np.float32)
        # Don't add noise at boundaries
        noise[:, 0, :] = 0.0
        noise[:, -1, :] = 0.0
        res = res + noise

    # Boundary-lock — enforce zero residual at start/goal
    res[:, 0, :] = 0.0
    res[:, -1, :] = 0.0

    bl = np.zeros((J,T,3), dtype=np.float32)
    for j in range(J):
        bl[j] = baseball_deform(bases[j], s[j], g[j])

    rot = (bl + res).transpose(1,0,2)
    full = rot.reshape(T, 72)
    if len(trans) != T:
        idx = np.linspace(0, len(trans)-1, T, dtype=int)
        trans = trans[idx]
    return fk_smpl_full(full, trans), rot


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-base", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fk-weight", type=float, default=30.0)
    p.add_argument("--animate", action="store_true", default=False, help="Generate animation GIF")
    p.add_argument("--scale", type=float, default=1.0, help="Residual amplitude scale (1.30 = Gen/True fix)")
    p.add_argument("--save-model", type=str, default=None, help="Save trained model to this path")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("[Stage5E-LSTM] Loading...")
    data, J = load_data(args.data_dir, args.n_files, args.n_frames, seed=args.seed)

    rng = np.random.default_rng(args.seed+1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:args.n_base]]
    train_data = data[perm[args.n_base:]][:int(0.8*(len(data)-args.n_base))]
    test_data = data[perm[args.n_base:]][int(0.8*(len(data)-args.n_base)):]

    bases = compute_mean_base(base_data[:,:,:72], args.n_base)
    model = BaseballVertexMDN(n_joints=J, n_timesteps=args.n_frames).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    best_val, xs, ys = train_model(model, train_data, bases, args.n_epochs,
                                    args.batch_size, args.lr, args.device,
                                    fk_loss_weight=args.fk_weight)
    print(f"Done {time.time()-t0:.0f}s best_val={best_val:.6f}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved: {args.save_model}")

    if len(test_data):
        test = test_data[0]
        rt = test[:args.n_frames,:72].reshape(args.n_frames,J,3)
        tr = test[:args.n_frames,72:75]
        g3d, gen_rot = generate_motion(model, bases, rt[0], rt[-1], tr, xs, ys, args.device,
                                        residual_scale=args.scale)
        t3d = fk_smpl_full(rt.reshape(args.n_frames,72), tr)
        rmse = float(np.sqrt(np.mean((g3d-t3d)**2)))
        gen_amp = float(np.std(gen_rot, axis=0).mean())
        true_amp = float(np.std(rt, axis=0).mean())
        print(f"RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}")

        # Align walking direction to X-axis (fix diagonal walking)
        # Use TRUE pelvis heading to align both motions to same direction
        n_head = max(10, args.n_frames // 5)
        head_t = t3d[n_head, 0, :2] - t3d[0, 0, :2]  # true pelvis XY heading
        ang = math.atan2(head_t[1], head_t[0])
        cos_a, sin_a = math.cos(-ang), math.sin(-ang)
        def rotate_xy(pos):
            rx = pos[..., 0] * cos_a - pos[..., 1] * sin_a
            ry = pos[..., 0] * sin_a + pos[..., 1] * cos_a
            out = pos.copy(); out[..., 0] = rx; out[..., 1] = ry
            return out
        g3d = rotate_xy(g3d)
        t3d = rotate_xy(t3d)
        tx = t3d[:, 0, 0]; wc = (tx.max() + tx.min()) / 2; FX = (tx.max() - tx.min()) / 2 + 1.5

        if args.animate:
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation, PillowWriter
                BONES=[(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
                       (9,13),(13,16),(16,18),(18,20),(20,22),(9,14),(14,17),(17,19),(19,21),(21,23)]
                OFF=np.array([0,-1.5,0])
                fig=plt.figure(figsize=(16,8)); ax=fig.add_subplot(111,projection='3d')
                def upd(f):
                    ax.cla()
                    xx,yy=np.meshgrid(np.linspace(wc-FX,wc+FX,15),np.linspace(-1.5,1.5,9))
                    ax.plot_surface(xx,yy,np.zeros_like(xx),alpha=0.06,color='gray')
                    jg=g3d[f]; jt=t3d[f]+OFF
                    for a,b in BONES:
                        ax.plot([jg[a,0],jg[b,0]],[jg[a,1],jg[b,1]],[jg[a,2],jg[b,2]],color='#1565C0',lw=3.0,alpha=0.9)
                        ax.plot([jt[a,0],jt[b,0]],[jt[a,1],jt[b,1]],[jt[a,2],jt[b,2]],color='#E53935',lw=3.0,alpha=0.9)
                    ax.set_xlim(wc-FX,wc+FX); ax.set_ylim(-1.5,1.5); ax.set_zlim(0,2.0); ax.view_init(elev=20,azim=-70)
                    ax.set_title(f'Stage5E-fixed | RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}',fontsize=13)
                anim=FuncAnimation(fig,upd,frames=args.n_frames,interval=100)
                out=Path(f"outputs/stage5e_fixed_{args.n_epochs}ep.gif"); out.parent.mkdir(parents=True,exist_ok=True)
                anim.save(str(out),writer=PillowWriter(fps=10)); plt.close()
                print(f"Animation: {out}")
            except Exception as e:
                print(f"Animation skipped: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
