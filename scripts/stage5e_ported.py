#!/usr/bin/env python3
"""
Stage5E-ported — stage5e wrapped in standard stage5 interface.

Original: stage5e_per_joint_10base_mdn.py
Standard interface: load_data, compute_mean_base, baseball_deform,
                    BaseballVertexMDN, train_model, generate_motion
"""

import sys, numpy as np, torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from smpl_fk import fk_smpl_full

# Import original stage5e implementation (no changes to original file)
from stage5e_per_joint_10base_mdn import (
    load_amass_pose_body,
    compute_per_joint_mean_bases,
    deform_path_to_start_and_goal,
    PerJointRotationMDN,
    StandardScalerNP,
    OneShotJointMDN,
    train_model as _train_model_original,
    generate_motion as _generate_motion_original,
)
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════
# Standard interface aliases
# ════════════════════════════════════════════════════════════

load_data = load_amass_pose_body
compute_mean_base = compute_per_joint_mean_bases
baseball_deform = deform_path_to_start_and_goal
BaseballVertexMDN = PerJointRotationMDN  # alias


# ════════════════════════════════════════════════════════════
# Standard train_model — returns (best_val, x_scaler, y_scaler)
# ════════════════════════════════════════════════════════════

def train_model(model, data75, bases, epochs=200, bs=16, lr=1e-3, device="cpu",
                aux_weights=None, winner_realism=None):
    """Standard wrapper: returns (best_val, x_scaler, y_scaler)."""
    return _train_model_original(model, data75, bases, epochs, bs, lr, device)


# ════════════════════════════════════════════════════════════
# Factory: create model with 24 joints (original defaults to 21)
# ════════════════════════════════════════════════════════════

def create_model(n_joints=24, n_timesteps=96, n_components=4, feature_dim=32, hidden_dim=128):
    """Create PerJointRotationMDN with given parameters."""
    return PerJointRotationMDN(
        n_joints=n_joints, n_timesteps=n_timesteps,
        n_components=n_components, feature_dim=feature_dim, hidden_dim=hidden_dim
    )


# ════════════════════════════════════════════════════════════
# Standard generate_motion — returns (joints_3d, rotation)
# ════════════════════════════════════════════════════════════

def generate_motion(model, bases, start, goal, trans,
                    x_scaler=None, y_scaler=None, device="cpu"):
    """Standard wrapper: x_scaler/y_scaler ignored (5e uses internal scalers)."""
    return _generate_motion_original(model, bases, start, goal, trans, device=device)


# ════════════════════════════════════════════════════════════
# Main — can be run standalone or via run_stage5_animate.py
# ════════════════════════════════════════════════════════════

def main():
    import argparse, time
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
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[Stage5E-ported] Loading...")
    data, J = load_data(args.data_dir, args.n_files, args.n_frames, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(data))
    base_data = data[perm[:args.n_base]]
    train_data = data[perm[args.n_base:]][:int(0.8 * (len(data) - args.n_base))]
    test_data = data[perm[args.n_base:]][int(0.8 * (len(data) - args.n_base)):]

    bases = compute_mean_base(base_data[:, :, :72], args.n_base)

    model = BaseballVertexMDN(n_joints=J, n_timesteps=args.n_frames).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    result = train_model(model, train_data, bases, args.n_epochs, args.batch_size, args.lr, args.device)

    # Handle different return types
    if isinstance(result, tuple) and len(result) == 3:
        best_val, xs, ys = result
    else:
        best_val = result
        xs = ys = None

    print(f"Done {time.time()-t0:.0f}s best_val={best_val:.6f}")

    if len(test_data):
        test = test_data[0]
        rt = test[:args.n_frames, :72].reshape(args.n_frames, J, 3)
        tr = test[:args.n_frames, 72:75]
        g3d, gen_rot = generate_motion(model, bases, rt[0], rt[-1], tr, xs, ys, args.device)
        t3d = fk_smpl_full(rt.reshape(args.n_frames, 72), tr)
        rmse = float(np.sqrt(np.mean((g3d - t3d) ** 2)))
        print(f"RMSE: {rmse:.4f}m")

        # Animation
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
            BONES=[(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
                   (9,13),(13,16),(16,18),(18,20),(20,22),(9,14),(14,17),(17,19),(19,21),(21,23)]
            OFF=np.array([0,-0.75,0])
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
                ax.set_title(f'Stage5E-ported | BLUE=Gen RED=True | RMSE={rmse:.4f}m',fontsize=13)
            anim=FuncAnimation(fig,upd,frames=args.n_frames,interval=100)
            out=Path(f"outputs/stage5e_ported_{args.n_epochs}ep.gif"); out.parent.mkdir(parents=True,exist_ok=True)
            anim.save(str(out),writer=PillowWriter(fps=10)); plt.close()
            print(f"Animation: {out}")
        except Exception as e:
            print(f"Animation skipped: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
