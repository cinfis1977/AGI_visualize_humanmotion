#!/usr/bin/env python3
"""
build_mini_amass.py — Mini AMASS Dataset Builder (v3, index-based)
==================================================================
Fast version using pre-built file index for O(1) feat_p resolution.

Usage:
  python build_mini_amass.py
  python build_mini_amass.py --max-per-action 200
  python build_mini_amass.py --target-frames 64 --max-per-action 50
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BABEL_DIR = DATA_DIR / "babel_labels"
AMASS_DIR = DATA_DIR / "amass_npz"
MINI_DIR = DATA_DIR / "mini_amass"

TARGET_FRAMES = 64
MAX_PER_ACTION = None

ACTION_MAP = {
    "walk": "walk", "run": "run", "jog": "jog", "jump": "jump",
    "hop": "jump", "leap": "jump",
    "sit": "sit", "stand": "stand",
    "bend": "bend", "crouch": "crouch", "squat": "squat",
    "turn": "turn", "fall": "fall",
    "kick": "kick", "throw": "throw", "crawl": "crawl",
}


# ── FILE INDEX ──────────────────────────────────────────────

def build_file_index():
    """Build mapping: (dataset, subject, stem) -> relative path."""
    print("[INDEX] Building AMASS file index...")
    t0 = time.time()
    index = {}
    for d in sorted(AMASS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for npz_file in d.rglob("*.npz"):
            rel = str(npz_file.relative_to(AMASS_DIR)).replace("\\", "/")
            name = npz_file.name
            for suf in ["_stageii.npz", "_stagei.npz", "_poses.npz"]:
                if name.endswith(suf):
                    name = name[:-len(suf)]
                    break
            subject = npz_file.parent.name
            key = f"{d.name}/{subject}/{name}"
            index[key] = rel
            # Also index by filename-only for loose matching
            index[f"__ANY__/{name}"] = rel
    print(f"[INDEX] {len(index)} entries in {time.time()-t0:.1f}s")
    return index


def resolve_feat_p(feat_p, index):
    """Convert BABEL feat_p to actual relative path using index."""
    parts = feat_p.split("/")
    dataset = parts[0]
    fname = parts[-1]

    for suf in ["_poses.npz", "_stageii.npz", "_stagei.npz"]:
        if fname.endswith(suf):
            fname = fname[:-len(suf)]
            break

    subject = parts[-2] if len(parts) >= 3 else ""

    # Try exact
    key = f"{dataset}/{subject}/{fname}"
    if key in index:
        return index[key]

    # Try without double prefix (CMU/CMU/86 -> CMU/86)
    if len(parts) >= 3 and parts[0] == parts[1]:
        key2 = f"{dataset}/{parts[-2]}/{fname}"
        if key2 in index:
            return index[key2]

    # Try second-to-last parent as subject
    if len(parts) >= 4 and parts[1] == parts[0]:
        key3 = f"{dataset}/{parts[-2]}/{fname}"
        if key3 in index:
            return index[key3]

    # Fallback: any subject in this dataset
    for k, v in index.items():
        if k.endswith(f"/{fname}") and k.startswith(f"{dataset}/"):
            return v

    return None


# ── BABEL PARSING ───────────────────────────────────────────

def map_to_canonical(act_cats):
    for cat in act_cats:
        cat_lower = cat.lower().strip()
        for keyword, canonical in ACTION_MAP.items():
            if keyword in cat_lower:
                return canonical
    return "other"


def extract_segments(ann):
    segments = []
    dur = ann.get("dur", 10.0)

    def add(seg, default_start=0.0, default_end=None):
        if not seg.get("act_cat"):
            return
        end = seg.get("end_t", default_end)
        if end is None:
            return
        segments.append({
            "act_cat": seg["act_cat"],
            "start_t": seg.get("start_t", default_start),
            "end_t": end,
            "seg_id": seg.get("seg_id", ""),
            "raw_label": seg.get("raw_label", ""),
        })

    for key in ["seq_ann", "frame_ann"]:
        ann_obj = ann.get(key)
        if ann_obj and ann_obj.get("labels"):
            for seg in ann_obj["labels"]:
                add(seg, default_start=0.0,
                    default_end=dur if key == "seq_ann" else None)

    for key in ["seq_anns", "frame_anns"]:
        anns = ann.get(key)
        if anns:
            for a in anns:
                if a.get("labels"):
                    for seg in a["labels"]:
                        add(seg, default_start=0.0, default_end=dur)

    return segments


def resample(arr, target_frames):
    """Linear resample (T, D) array to target_frames."""
    T = len(arr)
    if T <= 1:
        return np.tile(arr[:1], (target_frames, 1)) if len(arr) > 0 else arr
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_frames)
    result = np.zeros((target_frames, arr.shape[1]), dtype=np.float32)
    for i in range(arr.shape[1]):
        result[:, i] = np.interp(new_t, old_t, arr[:, i].astype(np.float32))
    return result


# ── MAIN PIPELINE ───────────────────────────────────────────

def main():
    global TARGET_FRAMES, MAX_PER_ACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--max-per-action", type=int, default=None)
    args = parser.parse_args()

    TARGET_FRAMES = args.target_frames
    MAX_PER_ACTION = args.max_per_action

    print("=" * 60)
    print("Mini AMASS Dataset Builder v3")
    print("=" * 60)
    print(f"Target frames: {TARGET_FRAMES}")
    print(f"Max per action: {MAX_PER_ACTION or 'unlimited'}")

    # Load data
    with open(BABEL_DIR / "filtered_actions.json") as f:
        filtered = json.load(f)
    print(f"Filtered sequences: {len(filtered)}")

    # Build index
    index = build_file_index()

    # Resolve feat_p
    print("[RESOLVE] Mapping BABEL feat_p to AMASS files...")
    t0 = time.time()
    seq_to_npz = {}
    missing_count = 0
    for sid, ann in filtered.items():
        fp = ann.get("feat_p", "")
        res = resolve_feat_p(fp, index)
        if res:
            seq_to_npz[sid] = AMASS_DIR / res
        else:
            missing_count += 1
    print(f"[RESOLVE] {len(seq_to_npz)}/{len(filtered)} resolved, "
          f"{missing_count} missing ({time.time()-t0:.1f}s)")

    # Process
    MINI_DIR.mkdir(parents=True, exist_ok=True)
    per_action = defaultdict(int)
    all_samples = []
    skipped_short = 0
    skipped_limit = 0

    print("[BUILD] Extracting motion segments...")
    t0 = time.time()

    for idx, (sid, ann) in enumerate(filtered.items()):
        if sid not in seq_to_npz:
            continue

        npz_path = seq_to_npz[sid]
        try:
            npz_data = dict(np.load(npz_path, allow_pickle=True))
        except Exception:
            continue

        poses = npz_data.get("poses", None)
        betas = npz_data.get("betas", None)
        trans = npz_data.get("trans", None)
        if poses is None:
            continue

        fps = float(npz_data.get("mocap_framerate", 30.0))
        n_total = len(poses)
        segments = extract_segments(ann)

        for seg in segments:
            canonical = map_to_canonical(seg["act_cat"])
            if canonical == "other":
                continue

            if MAX_PER_ACTION and per_action[canonical] >= MAX_PER_ACTION:
                skipped_limit += 1
                continue

            sf = max(0, min(int(seg["start_t"] * fps), n_total - 1))
            ef = max(sf + 1, min(int(seg["end_t"] * fps), n_total))
            n_seg = ef - sf
            if n_seg < 10:
                skipped_short += 1
                continue

            seg_poses = poses[sf:ef].astype(np.float32)
            seg_trans = trans[sf:ef].astype(np.float32) if trans is not None else np.zeros((n_seg, 3), dtype=np.float32)
            seg_betas = betas.astype(np.float32) if betas is not None else np.zeros(16, dtype=np.float32)

            poses_rs = resample(seg_poses, TARGET_FRAMES)
            trans_rs = resample(seg_trans, TARGET_FRAMES)

            sample = {
                "sequence_id": str(sid),
                "action_label": canonical,
                "dataset": ann.get("feat_p", "").split("/")[0],
                "feat_p": ann.get("feat_p", ""),
                "seg_id": seg["seg_id"],
                "raw_label": seg["raw_label"],
                "start_t": float(seg["start_t"]),
                "end_t": float(seg["end_t"]),
                "duration_sec": float(seg["end_t"] - seg["start_t"]),
                "fps": float(fps),
                "n_orig_frames": n_seg,
                "target_frames": TARGET_FRAMES,
                "poses": poses_rs,
                "betas": seg_betas,
                "trans": trans_rs,
            }

            sid_short = f"{canonical}_{sid}_{seg['seg_id'][:8]}"
            np.savez_compressed(MINI_DIR / f"{sid_short}.npz", **sample)

            all_samples.append({
                "sequence_id": str(sid),
                "action_label": canonical,
                "dataset": sample["dataset"],
                "feat_p": ann.get("feat_p", ""),
                "start_t": float(seg["start_t"]),
                "end_t": float(seg["end_t"]),
                "duration_sec": float(seg["end_t"] - seg["start_t"]),
                "n_orig_frames": n_seg,
                "file": f"{sid_short}.npz",
            })
            per_action[canonical] += 1

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(filtered)} seqs, "
                  f"{len(all_samples)} samples, {rate:.0f} seq/s")

    # ── Index ─────────────────────────────────────────────
    index_out = {
        "total_samples": len(all_samples),
        "actions": dict(sorted(per_action.items(), key=lambda x: -x[1])),
        "skipped_short": skipped_short,
        "skipped_limit": skipped_limit,
        "target_frames": TARGET_FRAMES,
        "missing_npz": missing_count,
        "samples": all_samples,
    }
    with open(MINI_DIR / "index.json", "w") as f:
        json.dump(index_out, f, indent=2)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Samples             : {len(all_samples)}")
    print(f"Skipped (short)     : {skipped_short}")
    print(f"Skipped (limit)     : {skipped_limit}")
    print(f"Missing npz         : {missing_count}")
    total = sum(per_action.values())
    for action, count in sorted(per_action.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct)
        print(f"  {action:<12s}: {count:>6d} ({pct:>5.1f}%) {bar}")
    print(f"\nSaved: {MINI_DIR}")


if __name__ == "__main__":
    main()
