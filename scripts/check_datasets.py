import os
from pathlib import Path

root = Path("c:/Dropbox/projects/AGI_visualize_humanmotion/data/amass_npz")
total_npz = 0
total_gb = 0

for d in sorted(root.iterdir()):
    if d.is_dir():
        npz_files = list(d.rglob("*.npz"))
        n = len(npz_files)
        size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        total_npz += n
        total_gb += size_gb
        print(f"{d.name:<20s}: {n:>5d} npz, {size_gb:>7.2f} GB")

print(f"{'TOTAL':<20s}: {total_npz:>5d} npz, {total_gb:>7.2f} GB")
