#!/usr/bin/env python3
"""
Stage5E-Colab — Google Colab optimized training.
CUDA GPU, differentiable FK, full-batch FK loss, foot-weighted.

Colab setup:
  1. Upload this file + smpl_fk_torch.py + smpl_fk.py to Colab
  2. Mount Google Drive with AMASS data: /content/drive/MyDrive/amass_npz/
  3. Or set --data-dir to your data path

Run:
  python stage5e_colab.py --epochs 100 --animate --scale 1.30
"""

import argparse, math, sys, time, os, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

# ════════════════════════════════════════════════════════════
# SMPL FK (DirectML-safe, CUDA-compatible)
# ════════════════════════════════════════════════════════════

# SMPL kinematic tree
KINTREE = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
           12, 13, 14, 16, 17, 18, 19, 20, 21]

# Bone vectors from SMPL neutral template (hardcoded for portability)
BONE_VECTORS = np.array([
    [ 0.000000,  0.000000,  0.000000],
    [ 0.069525, -0.091409, -0.006905],
    [-0.069525, -0.090748, -0.004369],
    [-0.004328, -0.114370,  0.001523],
    [ 0.102001, -0.689938,  0.016908],
    [-0.107756, -0.696424,  0.015049],
    [ 0.001159,  0.020810,  0.002615],
    [ 0.088406, -1.087899, -0.026785],
    [-0.091982, -1.094839, -0.027263],
    [ 0.002616,  0.073732,  0.028040],
    [ 0.114764, -1.143690,  0.092503],
    [-0.117354, -1.142983,  0.096085],
    [-0.000162,  0.287603, -0.014817],
    [ 0.081461,  0.195482, -0.006050],
    [-0.079143,  0.192565, -0.010575],
    [ 0.004990,  0.352572,  0.036532],
    [ 0.172438,  0.225951, -0.014918],
    [-0.175155,  0.225116, -0.019719],
    [ 0.432050,  0.213179, -0.042374],
    [-0.428897,  0.211787, -0.041119],
    [ 0.681284,  0.222165, -0.043545],
    [-0.684196,  0.219560, -0.046679],
    [ 0.765326,  0.214003, -0.058491],
    [-0.768817,  0.213442, -0.056994]
], dtype=np.float32)


def axis_angle_to_rotmat(aa):
    """Batched Rodrigues: (...,3) → (...,3,3). DirectML/CUDA safe."""
    theta = torch.sqrt((aa*aa).sum(dim=-1, keepdim=True) + 1e-12)
    k = aa / theta
    kx, ky, kz = k[...,0], k[...,1], k[...,2]
    z = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([z, -kz, ky], dim=-1),
        torch.stack([kz, z, -kx], dim=-1),
        torch.stack([-ky, kx, z], dim=-1),
    ], dim=-2)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    eye = aa.new_zeros(3, 3)
    eye[0,0]=1; eye[1,1]=1; eye[2,2]=1
    return eye + sin_t*K + (1.0-cos_t)*(K@K)


def fk_smpl_torch(pose_body, root_orient, trans):
    """Differentiable FK. (..., T, 63/3/3) → (..., T, 24, 3)."""
    lead = pose_body.shape[:-1]
    N = int(np.prod(lead)) if len(lead) else 1
    device, dtype = pose_body.device, pose_body.dtype

    pb = pose_body.reshape(N, 63)
    ro = root_orient.reshape(N, 3)
    tr = trans.reshape(N, 3)

    full = torch.zeros(N, 72, dtype=dtype, device=device)
    full[:, :3] = ro
    full[:, 3:66] = pb
    R = axis_angle_to_rotmat(full.reshape(N, 24, 3))

    bones = torch.as_tensor(BONE_VECTORS, dtype=dtype, device=device)

    world_R = [None]*24; world_p = [None]*24
    world_R[0] = R[:, 0]; world_p[0] = tr
    for j in range(1, 24):
        p = KINTREE[j]
        world_p[j] = world_p[p] + (world_R[p] @ bones[j].unsqueeze(-1)).squeeze(-1)
        world_R[j] = world_R[p] @ R[:, j]

    joints = torch.stack(world_p, dim=1)
    return joints.reshape(*lead, 24, 3)


# ════════════════════════════════════════════════════════════
# Data loader
# ════════════════════════════════════════════════════════════

def load_data(data_dir, n_files=200, n_frames=96, min_translation=1.0, seed=42, subject=None):
    base = Path(data_dir)
    if subject:
        # Only load from specific subject subdirectory
        base = base / subject
        print(f"Loading subject {subject} from {base}")
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
    N, T, D = data_rot.shape; J = D//3
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
# Model (OneShot MDN — same as stage5e)
# ════════════════════════════════════════════════════════════

class OneShotJointMDN(nn.Module):
    def __init__(self, feature_dim, n_timesteps=96, n_components=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        K = n_components
        self.logits = nn.Linear(hidden_dim, K)
        self.mu = nn.Linear(hidden_dim, K*n_timesteps*3)
        nn.init.zeros_(self.mu.bias)
        nn.init.normal_(self.mu.weight, 0, 0.01)

    def forward(self, x):
        h = self.net(x)
        K = self.logits.out_features
        return self.logits(h), self.mu(h).view(-1, K, self.mu.out_features//(K*3), 3)


class BaseballVertexMDN(nn.Module):
    def __init__(self, n_joints=24, n_timesteps=96, n_components=4, feature_dim=32, hidden_dim=128):
        super().__init__()
        self.n_joints = n_joints; self.n_timesteps = n_timesteps
        self.joint_encoder = nn.Sequential(nn.Linear(15, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.mdns = nn.ModuleList([
            OneShotJointMDN(feature_dim*2, n_timesteps, n_components, hidden_dim)
            for _ in range(n_joints)
        ])
        smooth_in = n_joints*n_components*3
        self.smooth = nn.Conv1d(smooth_in, smooth_in, kernel_size=3, padding=1, groups=n_joints, bias=False)
        nn.init.dirac_(self.smooth.weight)

    def forward(self, start_rot, goal_rot, action, base_features, apply_smooth=False):
        B, J = start_rot.shape[:2]; K = self.mdns[0].logits.out_features; T = self.n_timesteps
        action_exp = action.unsqueeze(1).expand(B, J, 3)
        feat = torch.cat([start_rot, goal_rot, base_features, action_exp], dim=-1)
        encoded = self.joint_encoder(feat)
        context, _ = self.attn(encoded, encoded, encoded)
        joint_input = torch.cat([encoded, context], dim=-1)
        all_logits, all_mu = [], []
        for j in range(J):
            l, m = self.mdns[j](joint_input[:,j,:])
            all_logits.append(l.unsqueeze(1)); all_mu.append(m.unsqueeze(1))
        logits = torch.cat(all_logits, 1)
        mu = torch.cat(all_mu, 1)

        if apply_smooth:
            Bs = mu.shape[0]
            mu_r = mu.view(Bs, J, K, T, 3).permute(0,1,2,4,3).reshape(Bs, J*K*3, T)
            mu_s = self.smooth(mu_r)
            mu = mu_s.view(Bs, J, K, 3, T).permute(0,1,2,4,3).contiguous().view(Bs, J, K, T, 3)
        return logits, mu


# ════════════════════════════════════════════════════════════
# Training — differentiable FK loss on FULL BATCH
# ════════════════════════════════════════════════════════════

def train_model(model, data75, bases, epochs=200, bs=16, lr=1e-3, device="cuda",
                fk_loss_weight=30.0):
    N, T, D = data75.shape; J = model.n_joints
    rot_data = data75[:,:,:72].reshape(N, T, J, 3)
    trans_data = data75[:,:,72:75]
    R = torch.tensor(rot_data, dtype=torch.float32, device=device)
    Tr = torch.tensor(trans_data, dtype=torch.float32, device=device)

    print("  Computing baselines...")
    bl = np.zeros((N, T, J, 3), dtype=np.float32)
    for n in range(N):
        for j in range(J):
            bl[n,:,j] = baseball_deform(bases[j], rot_data[n,0,j], rot_data[n,-1,j])
    BL = torch.tensor(bl, dtype=torch.float32, device=device)

    res = R.permute(0,2,1,3) - BL.permute(0,2,1,3)
    Y = res.reshape(N, J, -1)

    sg = np.concatenate([rot_data[:,0], rot_data[:,-1]], -1).reshape(N, J, 6)
    sg_mean = sg.reshape(-1,6).mean(0).astype(np.float32)
    sg_std = sg.reshape(-1,6).std(0).astype(np.float32)+1e-8

    class S:
        def __init__(self,m,s): self.mean=m; self.std=s
        def transform(self,X): return (X-self.mean)/self.std
        def inverse(self,X): return X*self.std+self.mean
    xs = S(sg_mean, sg_std)

    sg_s = xs.transform(sg.reshape(-1,6)).reshape(N, J, 6)
    Ss = torch.tensor(sg_s[...,:3], dtype=torch.float32, device=device)
    Gs = torch.tensor(sg_s[...,3:], dtype=torch.float32, device=device)

    base_feat = np.zeros((N, J, 6), dtype=np.float32)
    for j in range(J):
        bj = bases[j]
        base_feat[:,j,0] = np.linalg.norm(bj[1:]-bj[:-1], axis=-1).sum()
        base_feat[:,j,1] = np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2], axis=-1).mean() if T>2 else 0
        base_feat[:,j,3] = bj[:,2].max()
        base_feat[:,j,4] = bj[:,2].std()
        base_feat[:,j,5] = bj[-1,2]-bj[0,2]
    Bf = torch.tensor(base_feat, dtype=torch.float32, device=device)
    actions = torch.zeros(N, 3, device=device); actions[:,0]=1.0

    nv = max(1, int(0.15*N))
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    p = torch.randperm(N, generator=g, device="cpu")
    vi, ti = p[:nv], p[nv:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf"); best_state = None
    K = 4

    # Per-joint FK weights: feet 10x, ankles 8x, knees 3x
    FK_W = torch.ones(J, device=device)
    FK_W[7]=8; FK_W[8]=8; FK_W[10]=10; FK_W[11]=10; FK_W[4]=3; FK_W[5]=3

    print(f"  Colab: diff-FK full-batch + foot-weights | train={len(ti)} val={nv} bs={bs}")

    for ep in range(1, epochs+1):
        model.train()
        tp = torch.randperm(len(ti), device="cpu")
        total, cnt, fk_sum, amp_sum = 0.0, 0, 0.0, 0.0
        for st in range(0, len(ti), bs):
            b = ti[tp[st:st+bs]]
            logits, mu = model(Ss[b], Gs[b], actions[b], Bf[b], apply_smooth=True)
            y = Y[b].view(-1, J, T, 3).unsqueeze(2)  # (B,J,1,T,3)

            # MCL loss
            diff = y - mu
            err = (diff**2).mean((-2,-1))
            win = err.argmin(-1)
            w = torch.full_like(err, 0.05/(K-1))
            for bi in range(win.shape[0]):
                for ji in range(J):
                    w[bi, ji, win[bi, ji]] = 0.95
            recon = (w*err).sum(-1).mean()
            gate = sum(F.cross_entropy(logits[:,j], win[:,j]) for j in range(J))/J

            # === Differentiable FK loss on FULL BATCH ===
            B_cur = len(b)
            mu_win = mu.gather(2, win.view(B_cur,J,1,1,1).expand(B_cur,J,1,T,3)).squeeze(2)  # (B,J,T,3)
            pred_rot = BL[b] + mu_win.permute(0,2,1,3)  # (B,T,J,3)
            true_rot = R[b]

            # FK: position error in world space (differentiable!)
            pred_flat = pred_rot.reshape(B_cur, T, 72)
            true_flat = true_rot.reshape(B_cur, T, 72)
            pred_pos = fk_smpl_torch(pred_flat[:,:,3:66], pred_flat[:,:,:3], Tr[b])
            true_pos = fk_smpl_torch(true_flat[:,:,3:66], true_flat[:,:,:3], Tr[b])

            pos_err = ((pred_pos - true_pos)**2).sum(-1)  # (B, T, 24)
            fk_loss = (pos_err * FK_W.unsqueeze(0).unsqueeze(0)).mean()

            # Amplitude matching (detached, numpy)
            with torch.no_grad():
                pr = pred_rot.detach().cpu().numpy()
                tr_np = true_rot.detach().cpu().numpy()
                p_std = np.std(pr, axis=1).mean()  # mean over batch+T, per joint
                t_std = np.std(tr_np, axis=1).mean()
                amp_loss_t = torch.tensor((p_std-t_std)**2, dtype=torch.float32, device=device)

            loss = 100.0*recon + 0.1*gate + fk_loss_weight*fk_loss + 30.0*amp_loss_t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()*len(b); cnt += len(b)
            fk_sum += fk_loss.item()*len(b)
            amp_sum += amp_loss_t.item()*len(b)

        # Validation
        model.eval()
        with torch.no_grad():
            vl, vm = model(Ss[vi], Gs[vi], actions[vi], Bf[vi], apply_smooth=True)
            vd = Y[vi].view(-1,J,T,3).unsqueeze(2) - vm
            ve = (vd**2).mean((-2,-1))
            vw = ve.argmin(-1)
            vw2 = torch.full_like(ve, 0.05/(K-1))
            for bi in range(vw.shape[0]):
                for ji in range(J):
                    vw2[bi, ji, vw[bi, ji]] = 0.95
            val_loss = (vw2*ve).sum(-1).mean().item()

        if val_loss < best_val-1e-9:
            best_val = val_loss
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

        if ep==1 or ep%10==0 or ep==epochs:
            print(f"  ep {ep:4d}: train={total/max(cnt,1):.4f} val={val_loss:.6f} "
                  f"fk={fk_sum/max(cnt,1):.6f} amp={amp_sum/max(cnt,1):.6f} best={best_val:.6f}")

    if best_state: model.load_state_dict(best_state)
    return best_val, xs


# ════════════════════════════════════════════════════════════
# Generation
# ════════════════════════════════════════════════════════════

def generate_motion(model, bases, start, goal, trans, x_scaler, device, residual_scale=1.0):
    model.eval()
    J, T = 24, model.n_timesteps
    s = np.asarray(start, np.float32).reshape(J,3)
    g = np.asarray(goal, np.float32).reshape(J,3)

    sg = x_scaler.transform(np.concatenate([s,g],-1))
    S = torch.tensor(sg[:,:3], dtype=torch.float32, device=device).unsqueeze(0)
    G = torch.tensor(sg[:,3:], dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.zeros(1,3,device=device); action[:,0]=1.0

    bf = torch.zeros(1,J,6,device=device)
    for j in range(J):
        bj = bases[j]
        bf[0,j,0] = torch.tensor(np.linalg.norm(bj[1:]-bj[:-1],axis=-1).sum(),dtype=torch.float32)
        bf[0,j,1] = torch.tensor(np.linalg.norm(bj[2:]-2*bj[1:-1]+bj[:-2],axis=-1).mean() if T>2 else 0.0,dtype=torch.float32)
        bf[0,j,3] = torch.tensor(bj[:,2].max(),dtype=torch.float32)
        bf[0,j,4] = torch.tensor(bj[:,2].std(),dtype=torch.float32)
        bf[0,j,5] = torch.tensor(bj[-1,2]-bj[0,2],dtype=torch.float32)

    with torch.no_grad():
        logits, mu = model(S, G, action, bf, apply_smooth=True)
        k = logits[0].argmax(-1)
        mu_k = torch.stack([mu[0,j,k[j]] for j in range(J)])

    res = mu_k.cpu().numpy()
    if residual_scale != 1.0:
        res = res * residual_scale
    res[:,0,:] = 0.0; res[:,-1,:] = 0.0

    bl = np.zeros((J,T,3), dtype=np.float32)
    for j in range(J):
        bl[j] = baseball_deform(bases[j], s[j], g[j])

    rot = (bl + res).transpose(1,0,2)
    full = rot.reshape(T, 72)
    if len(trans) != T:
        idx = np.linspace(0,len(trans)-1,T,dtype=int)
        trans = trans[idx]

    # Use differentiable FK (already have it, no extra dependency needed)
    rot_t = torch.tensor(full, dtype=torch.float32, device=device)
    trans_t = torch.tensor(trans, dtype=torch.float32, device=device)
    joints = fk_smpl_torch(rot_t[:, 3:66].unsqueeze(0),
                           rot_t[:, :3].unsqueeze(0),
                           trans_t.unsqueeze(0)).squeeze(0)
    return joints.cpu().numpy(), rot


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content/drive/MyDrive/amass_npz")
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--n-base", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fk-weight", type=float, default=30.0)
    p.add_argument("--animate", action="store_true", default=False)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--subject", type=str, default=None, help="KIT subject ID (e.g. 675=119 walks, 9=70)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage5E-Colab] Device: {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, J = load_data(args.data_dir, args.n_files, args.n_frames, seed=args.seed, subject=args.subject)
    rng = np.random.default_rng(args.seed+1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:args.n_base]]
    train_data = data[perm[args.n_base:]][:int(0.8*(len(data)-args.n_base))]
    test_data = data[perm[args.n_base:]][int(0.8*(len(data)-args.n_base)):]

    bases = compute_mean_base(base_data[:,:,:72], args.n_base)
    model = BaseballVertexMDN(n_joints=J, n_timesteps=args.n_frames).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    best_val, xs = train_model(model, train_data, bases, args.epochs,
                                args.batch_size, args.lr, device,
                                fk_loss_weight=args.fk_weight)
    print(f"Done {time.time()-t0:.0f}s best_val={best_val:.6f}")

    if len(test_data):
        test = test_data[0]
        rt = test[:args.n_frames,:72].reshape(args.n_frames,J,3)
        tr = test[:args.n_frames,72:75]
        g3d, gen_rot = generate_motion(model, bases, rt[0], rt[-1], tr, xs, device, args.scale)

        # Ground truth FK via torch (self-contained, no smpl_fk.py needed)
        rt_t = torch.tensor(rt.reshape(args.n_frames,72), dtype=torch.float32, device=device)
        tr_t = torch.tensor(tr, dtype=torch.float32, device=device)
        t3d = fk_smpl_torch(rt_t[:, 3:66].unsqueeze(0),
                            rt_t[:, :3].unsqueeze(0),
                            tr_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        rmse = float(np.sqrt(np.mean((g3d-t3d)**2)))
        gen_amp = float(np.std(gen_rot, axis=0).mean())
        true_amp = float(np.std(rt, axis=0).mean())
        print(f"RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}")

        if args.animate:
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation, PillowWriter
                BONES=[(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
                       (9,13),(13,16),(16,18),(18,20),(20,22),(9,14),(14,17),(17,19),(19,21),(21,23)]
                OFF=np.array([0,-1.5,0])
                tx=t3d[:,0,0]; wc=(tx.max()+tx.min())/2; FX=(tx.max()-tx.min())/2+1.5
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
                    ax.set_title(f'Stage5E-Colab | RMSE={rmse:.4f}m Gen/True={gen_amp/max(true_amp,1e-8):.2f}',fontsize=13)
                anim=FuncAnimation(fig,upd,frames=args.n_frames,interval=100)
                out=Path(f"stage5e_colab_{args.epochs}ep.gif")
                anim.save(str(out),writer=PillowWriter(fps=10)); plt.close()
                print(f"Animation: {out}")
            except Exception as e:
                print(f"Animation skipped: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
