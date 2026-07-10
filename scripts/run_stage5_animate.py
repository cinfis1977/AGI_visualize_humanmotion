#!/usr/bin/env python3
"""
Generic stage5 training + animation runner.

Usage:
  python scripts/run_stage5_animate.py <module_name> [--n-epochs 60] [--n-base 10] [--device cpu]

Examples:
  python scripts/run_stage5_animate.py stage5p_clean_baseball_clean_natural_v4
  python scripts/run_stage5_animate.py stage5m_time_cross_attention_vertex_mdn --n-epochs 40
  python scripts/run_stage5_animate.py stage5e_per_joint_10base_mdn --n-base 20
"""
import argparse, sys, numpy as np, torch, time, importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from smpl_fk import fk_smpl_full

def main():
    p = argparse.ArgumentParser()
    p.add_argument("module", help="Stage5 module name (without .py)")
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--n-base", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--n-files", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-dir", default="C:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
    p.add_argument("--per-vertex", action="store_true", help="Use per-vertex independent 8-sample base")
    p.add_argument("--out", default=None, help="Output GIF path")
    args = p.parse_args()

    # Import the module
    s5p = importlib.import_module(args.module)
    J = 24

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Module: {args.module}")
    data, J = s5p.load_data(args.data_dir, args.n_files, args.n_frames, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(data))

    if args.per_vertex:
        base_pool = perm[:40]
        train_data = data[perm[40:]][:int(0.8*(len(data)-40))]
        test_data = data[perm[40:]][int(0.8*(len(data)-40)):]
        bases = np.zeros((J, args.n_frames, 3), dtype=np.float32)
        for j in range(J):
            pool_j = data[base_pool][:, :args.n_frames, j*3:(j+1)*3]
            picks = rng.choice(len(base_pool), 8, replace=False)
            bases[j] = np.nanmean(pool_j[picks], axis=0)
        print(f"Per-vertex independent base: 8 samples/joint from {len(base_pool)} pool")
    else:
        base_data = data[perm[:args.n_base]]
        train_data = data[perm[args.n_base:]][:int(0.8*(len(data)-args.n_base))]
        test_data = data[perm[args.n_base:]][int(0.8*(len(data)-args.n_base)):]
        bases = s5p.compute_mean_base(base_data[:,:,:72], args.n_base)
        print(f"Shared base: {args.n_base} samples")

    # Try standard constructor first, fall back to stage5e-style
    try:
        model = s5p.BaseballVertexMDN(J=J, T=args.n_frames, K=4, D=64, H=128).to(args.device)
    except TypeError:
        # Check for factory function first
        if hasattr(s5p, 'create_model'):
            model = s5p.create_model(n_joints=J, n_timesteps=args.n_frames).to(args.device)
        else:
            model = s5p.BaseballVertexMDN(n_joints=J, n_timesteps=args.n_frames).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    result = s5p.train_model(model, train_data, bases, args.n_epochs,
                                         args.batch_size, args.lr, args.device)
    # Handle different return types across stage5 versions
    if isinstance(result, tuple):
        if len(result) >= 4:
            best_val, baselines_t, xs, ys = result[0], result[1], result[2], result[3]
        elif len(result) == 3:
            best_val, xs, ys = result
        elif len(result) == 2:
            best_val, xs = result; ys = None
        else:
            best_val = result[0]; xs = None; ys = None
    else:
        best_val = result; xs = None; ys = None
    print(f"Done {time.time()-t0:.0f}s best_val={best_val:.6f}")

    # Test
    test = test_data[0]
    rt = test[:args.n_frames,:72].reshape(args.n_frames, J, 3)
    tr = test[:args.n_frames, 72:75]
    # Try standard generate_motion, fall back to older signature
    try:
        result_gen = s5p.generate_motion(model, bases, rt[0], rt[-1], tr, xs, ys, args.device)
    except TypeError:
        result_gen = s5p.generate_motion(model, bases, rt[0], rt[-1], tr, device=args.device)
    if isinstance(result_gen, tuple) and len(result_gen) == 2:
        g3d, gen_rot = result_gen
    else:
        g3d = result_gen; gen_rot = None
    t3d = fk_smpl_full(rt.reshape(args.n_frames, 72), tr)
    rmse = float(np.sqrt(np.mean((g3d - t3d) ** 2)))
    if gen_rot is not None:
        gen_amp = float(np.std(gen_rot, axis=0).mean())
        true_amp = float(np.std(rt, axis=0).mean())
        extra = f"  Gen/True={gen_amp/max(true_amp,1e-8):.2f}"
    else:
        extra = ""
    print(f"RMSE={rmse:.4f}m{extra}")

    # Animation
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        BONES = [(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),(1,4),(4,7),(7,10),
                 (2,5),(5,8),(8,11),(9,13),(13,16),(16,18),(18,20),(20,22),
                 (9,14),(14,17),(17,19),(19,21),(21,23)]
        OFF = np.array([0, -0.75, 0])
        tx = t3d[:, 0, 0]
        wc = (tx.max() + tx.min()) / 2
        FX = (tx.max() - tx.min()) / 2 + 1.5
        FY, FZ = 1.5, 2.0

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')

        def upd(f):
            ax.cla()
            xx, yy = np.meshgrid(np.linspace(wc-FX, wc+FX, 15), np.linspace(-FY, FY, 9))
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color='gray')
            jg, jt = g3d[f], t3d[f] + OFF
            for a, b in BONES:
                ax.plot([jg[a,0],jg[b,0]], [jg[a,1],jg[b,1]], [jg[a,2],jg[b,2]],
                        color='#1565C0', lw=3.0, alpha=0.9)
                ax.plot([jt[a,0],jt[b,0]], [jt[a,1],jt[b,1]], [jt[a,2],jt[b,2]],
                        color='#E53935', lw=3.0, alpha=0.9)
            ax.set_xlim(wc-FX, wc+FX); ax.set_ylim(-FY, FY); ax.set_zlim(0, FZ)
            ax.view_init(elev=20, azim=-70)
            ax.set_title(f'{args.module} | BLUE=Gen RED=True | RMSE={rmse:.4f}m', fontsize=13)

        anim = FuncAnimation(fig, upd, frames=args.n_frames, interval=100)
        out_path = Path(args.out or f"outputs/{args.module}_{args.n_epochs}ep.gif")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out_path), writer=PillowWriter(fps=10))
        plt.close()
        print(f"Animation: {out_path}")
    except Exception as e:
        print(f"Animation skipped: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
