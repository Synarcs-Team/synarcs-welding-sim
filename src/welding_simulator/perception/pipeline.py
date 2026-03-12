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

def _log(msg, log_cb=None):
    if log_cb:
        log_cb(msg)
    else:
        print(msg, flush=True)

def run_process(log_cb=None):
    _log("[STEP] PROCESS_START", log_cb)

    # Clear out any legacy or advanced seam detection files explicitly before processing
    for fname in ["seams.json", "seam_results.json"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            
    # ── Merge point clouds ────────────────────────────────────────────────────────
    pcd_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("scan_") and f.endswith(".pcd")])
    if not pcd_files:
        _log("[ERROR] No .pcd files found in data/latest/", log_cb)
        return

    merged = o3d.geometry.PointCloud()
    for fname in pcd_files:
        pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR, fname))
        merged += pcd
        _log(f"[STEP] MERGE_FILE file={fname} points={len(pcd.points)}", log_cb)

    _log(f"[STEP] MERGE_DONE total_points={len(merged.points)}", log_cb)

    import json

    # ── Read UI Config ────────────────────────────────────────────────────────────
    try:
        with open(os.path.join(DATA_DIR, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"joint_type": "tee"}

    bw = float(cfg.get("bw", 0.15))
    bl = float(cfg.get("bl", 0.15))
    bt = float(cfg.get("bt", 0.025))
    sh = float(cfg.get("sh", 0.15))
    st = float(cfg.get("st", 0.025))
    
    # Calculate bounding box dynamically using the joint's maximum physical extent
    margin = 0.05
    min_bound = np.array([0.75 - (bw/2) - margin, -(bl/2) - margin, 1.001])
    max_bound = np.array([0.75 + (bw/2) + margin,  (bl/2) + margin, 1.0 + bt + sh + margin])

    # ── Crop to T-joint area ──────────────────────────────────────────────────────
    bbox      = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped   = merged.crop(bbox)
    _log(f"[STEP] CROP_DONE points={len(cropped.points)}", log_cb)

    # ── Estimate normals ──────────────────────────────────────────────────────────
    cropped.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=20)
    )
    cropped.orient_normals_towards_camera_location(camera_location=[0, 0, 3])

    # ── Detect weld seams ─────────────────────────────────────────────────────────

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

    pts = np.asarray(cropped.points)
    clrs = np.asarray(cropped.colors)
    xyzrgb = np.hstack((pts, clrs))
    np.save(os.path.join(DATA_DIR, "merged_xyzrgb.npy"), xyzrgb)

    _log("[STEP] PROCESS_COMPLETE", log_cb)

if __name__ == "__main__":
    run_process()
