#!/usr/bin/env python3
"""
BABEL Action Filter
-------------------
Reads BABEL label JSONs, finds sequences containing target actions,
and outputs:
  1. needed_amass_files.txt  — list of AMASS .npz files to download
  2. filtered_actions.json   — filtered BABEL entries with target actions only
  3. action_stats.json       — count of sequences per action category

Target actions (human motion imagination pipeline):
  walk, run, jog, jump, sit, stand, bend, crouch, squat,
  turn, fall, kick, throw

Usage:
  python filter_babel_actions.py
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
BABEL_DIR = Path("../data/babel_labels")
OUTPUT_DIR = Path("../data/babel_labels")
TARGET_ACTIONS = {
    # (search_term, canonical_name)
    "walk": "walk",
    "run": "run",
    "jog": "jog",
    "jump": "jump",
    "hop": "jump",       # hop -> jump olarak grupla
    "leap": "jump",      # leap -> jump
    "sit": "sit",
    "stand": "stand",
    "bend": "bend",
    "crouch": "crouch",
    "squat": "squat",
    "turn": "turn",
    "fall": "fall",
    "kick": "kick",
    "throw": "throw",
    "crawl": "crawl",
}

BABEL_FILES = ["train.json", "val.json", "test.json",
               "extra_train.json", "extra_val.json"]


def load_babel(filepath: Path) -> dict:
    """Load a BABEL JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_act_cats(ann: dict) -> set:
    """Extract all action categories (act_cat) from a BABEL annotation."""
    cats = set()

    # Sequence annotation
    seq_ann = ann.get("seq_ann")
    if seq_ann and seq_ann.get("labels"):
        for seg in seq_ann["labels"]:
            if seg.get("act_cat"):
                cats.update(seg["act_cat"])

    # Frame annotation
    frame_ann = ann.get("frame_ann")
    if frame_ann and frame_ann.get("labels"):
        for seg in frame_ann["labels"]:
            if seg.get("act_cat"):
                cats.update(seg["act_cat"])

    # Extra annotations (plural keys)
    seq_anns = ann.get("seq_anns")
    if seq_anns:
        for sann in seq_anns:
            if sann.get("labels"):
                for seg in sann["labels"]:
                    if seg.get("act_cat"):
                        cats.update(seg["act_cat"])

    frame_anns = ann.get("frame_anns")
    if frame_anns:
        for fann in frame_anns:
            if fann.get("labels"):
                for seg in fann["labels"]:
                    if seg.get("act_cat"):
                        cats.update(seg["act_cat"])

    return cats


def match_target_actions(act_cats: set) -> set:
    """Match BABEL action categories to our target actions."""
    matched = set()
    for cat in act_cats:
        cat_lower = cat.lower().strip()
        for keyword, canonical in TARGET_ACTIONS.items():
            if keyword in cat_lower:
                matched.add(canonical)
    return matched


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_filtered = {}          # {babel_sid: annotation}
    needed_amass = set()       # set of feat_p paths
    action_counter = Counter() # action -> count of sequences
    per_file_stats = {}        # {filename: {action: count}}

    for bfile in BABEL_FILES:
        fpath = BABEL_DIR / bfile
        if not fpath.exists():
            print(f"[SKIP] {fpath} not found — download BABEL first")
            continue

        print(f"[LOAD] {bfile}")
        data = load_babel(fpath)
        file_counter = Counter()
        file_filtered = {}

        for sid, ann in data.items():
            act_cats = extract_act_cats(ann)
            matched = match_target_actions(act_cats)

            if matched:
                file_filtered[sid] = ann

                # Track AMASS file needed
                feat_p = ann.get("feat_p", "")
                if feat_p:
                    needed_amass.add(feat_p)

                # Count
                for action in matched:
                    file_counter[action] += 1
                    action_counter[action] += 1

        all_filtered.update(file_filtered)
        per_file_stats[bfile] = dict(file_counter)
        print(f"  -> {len(file_filtered)} sequences matched")

    # ── Write needed AMASS files ────────────────────────────
    needed_path = OUTPUT_DIR / "needed_amass_files.txt"
    with open(needed_path, "w", encoding="utf-8") as f:
        for feat_p in sorted(needed_amass):
            f.write(feat_p + "\n")

    # Also group by dataset for easier download
    dataset_files = defaultdict(list)
    for feat_p in needed_amass:
        dataset_name = feat_p.split("/")[0]
        dataset_files[dataset_name].append(feat_p)

    needed_by_dataset_path = OUTPUT_DIR / "needed_amass_by_dataset.txt"
    with open(needed_by_dataset_path, "w", encoding="utf-8") as f:
        for dname in sorted(dataset_files.keys()):
            f.write(f"\n# -- {dname} ({len(dataset_files[dname])} files) --\n")
            for feat_p in sorted(dataset_files[dname]):
                f.write(feat_p + "\n")

    # ── Write filtered actions JSON ─────────────────────────
    filtered_path = OUTPUT_DIR / "filtered_actions.json"
    with open(filtered_path, "w") as f:
        json.dump(all_filtered, f, indent=2)

    # ── Write action stats ──────────────────────────────────
    stats = {
        "total_sequences": len(all_filtered),
        "total_unique_amass_files": len(needed_amass),
        "actions": dict(action_counter.most_common()),
        "per_file": per_file_stats,
        "datasets": {d: len(fs) for d, fs in dataset_files.items()},
    }
    stats_path = OUTPUT_DIR / "action_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total matched sequences : {len(all_filtered)}")
    print(f"Unique AMASS .npz files : {len(needed_amass)}")
    print(f"Datasets needed         : {len(dataset_files)}")
    print()
    print("Action distribution:")
    for action, count in action_counter.most_common():
        bar = "█" * (count // max(1, action_counter.most_common(1)[0][1] // 30))
        print(f"  {action:<12s} : {count:>5d}  {bar}")
    print()
    print("Per dataset:")
    for dname in sorted(dataset_files.keys()):
        print(f"  {dname:<20s} : {len(dataset_files[dname]):>5d} files")
    print()
    print(f"Output files written to: {OUTPUT_DIR}")
    print(f"  - needed_amass_files.txt")
    print(f"  - needed_amass_by_dataset.txt")
    print(f"  - filtered_actions.json")
    print(f"  - action_stats.json")


if __name__ == "__main__":
    main()
