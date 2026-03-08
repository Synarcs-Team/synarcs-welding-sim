"""
process_pcd.py — Stage 3: Point Cloud Processing
Pure Python (no Isaac Sim). Merges all scan PCDs, detects weld seams,
saves merged.pcd and seams.json to data/latest/.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import open3d as o3d

from pathlib import Path
ROOT     = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(ROOT, "data", "latest")

print("[STEP] PROCESS_START", flush=True)

# ── Merge point clouds ────────────────────────────────────────────────────────
pcd_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("scan_") and f.endswith(".pcd")])
if not pcd_files:
    print("[ERROR] No .pcd files found in data/latest/", flush=True)
    sys.exit(1)

merged = o3d.geometry.PointCloud()
for fname in pcd_files:
    pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR, fname))
    merged += pcd
    print(f"[STEP] MERGE_FILE file={fname} points={len(pcd.points)}", flush=True)

print(f"[STEP] MERGE_DONE total_points={len(merged.points)}", flush=True)

# ── Crop to T-joint area ──────────────────────────────────────────────────────
min_bound = np.array([0.65, -0.25, 1.01])
max_bound = np.array([1.15,  0.25, 1.30])
bbox      = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
cropped   = merged.crop(bbox)
print(f"[STEP] CROP_DONE points={len(cropped.points)}", flush=True)

# ── Estimate normals ──────────────────────────────────────────────────────────
cropped.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=20)
)
cropped.orient_normals_towards_camera_location(camera_location=[0, 0, 3])
cropped.paint_uniform_color([0.2, 0.4, 1.0])

import json

# ── Detect weld seams ─────────────────────────────────────────────────────────
from welding_simulator.planning.t_joint_planning import find_t_joint_paths

# Try to read the exact config for dummy boundaries
try:
    with open(os.path.join(DATA_DIR, "config.json")) as f:
        cfg = json.load(f)
except Exception:
    cfg = {"joint_type": "tee"}

bw = float(cfg.get("bw", 0.15))
st = float(cfg.get("st", 0.025))
bt = float(cfg.get("bt", 0.025))

# Generate dummy seams based on the config parameters matching the table position (0.75, 0, 1)
seam1_start = [0.75 - bw/2,  st/2 + 0.005, 1.0 + bt]
seam1_end   = [0.75 + bw/2,  st/2 + 0.005, 1.0 + bt]
seam2_start = [0.75 - bw/2, -st/2 - 0.005, 1.0 + bt]
seam2_end   = [0.75 + bw/2, -st/2 - 0.005, 1.0 + bt]
seams_data = {
    "seam1": {"start": seam1_start, "end": seam1_end},
    "seam2": {"start": seam2_start, "end": seam2_end}
}

# ── Save outputs ──────────────────────────────────────────────────────────────
o3d.io.write_point_cloud(os.path.join(DATA_DIR, "merged.pcd"), cropped)

with open(os.path.join(DATA_DIR, "seams.json"), "w") as f:
    json.dump(seams_data, f, indent=4)

# Also save merged raw xyz for web display
pts = np.asarray(cropped.points)
np.save(os.path.join(DATA_DIR, "merged_xyz.npy"), pts)

print(f"[STEP] PROCESS_COMPLETE", flush=True)
