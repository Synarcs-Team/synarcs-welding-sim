"""
adaptive_slice.py — AdaptiveSlice Seam Detection Algorithm (SEAM-LOC 009)

Wrapper around the orientation-independent PCA-histogram-slicing algorithm.
Exposes run_seam_detection(log_cb=None) to match the Triplane RANSAC interface
so the API can dispatch to either algorithm interchangeably.

Output files (same schema as seam_detector.py):
  data/latest/seam_results.json   — rich results
  data/latest/seams.json          — legacy start/end for 3D viewer
"""

import os
import json
import sys
import numpy as np
from pathlib import Path

# ── ensure the project root is in PYTHONPATH when run standalone ──────────────
ROOT     = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(ROOT, "data", "latest")

sys.stdout.reconfigure(line_buffering=True)


def _log(msg, log_cb=None):
    if log_cb:
        log_cb(msg)
    else:
        print(msg, flush=True)


# ── Internal copy of SEAM-LOC 009 core (avoids newalg.py import path issues) ─

import re
import time
import warnings
warnings.filterwarnings('ignore')


class SeamDetectionError(RuntimeError):
    ERROR_CODE = 'E000'

class IsolationError(SeamDetectionError):
    ERROR_CODE = 'E001'

class InsufficientPointsError(SeamDetectionError):
    ERROR_CODE = 'E002'

class InsufficientSlicesError(SeamDetectionError):
    ERROR_CODE = 'E003'


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def _projected_thinness(pts, base_normal):
    pp = pts - np.outer(pts @ base_normal, base_normal)
    ev, _ = np.linalg.eigh(np.cov(pp.T))
    ev = np.sort(ev)
    return ev[1] / (ev[2] + 1e-10)


def _fit_line_pca(corner_pts):
    anchor = np.mean(corner_pts, axis=0)
    ev, evec = np.linalg.eigh(np.cov(corner_pts.T))
    direction = evec[:, np.argmax(ev)]
    projs = corner_pts @ direction
    t_min, t_max = projs.min(), projs.max()
    t_mid = (t_min + t_max) / 2
    anchor_on_line = anchor + (t_mid - anchor @ direction) * direction
    start = anchor_on_line + (t_min - t_mid) * direction
    end   = anchor_on_line + (t_max - t_mid) * direction
    return anchor_on_line, direction, start, end


def _make_waypoints(start, end, spacing_mm=10.0):
    length = np.linalg.norm(end - start)
    n_wp   = max(2, int(length / spacing_mm) + 1)
    return np.array([start + t * (end - start) for t in np.linspace(0.0, 1.0, n_wp)])


def _detect(points, log_cb=None):
    """Core AdaptiveSlice T-joint detection. Returns rich result dict."""
    t_start = time.perf_counter()

    if len(points) < 100:
        raise InsufficientPointsError(f"Only {len(points)} points — minimum 100 required.")

    _log("[INFO] Step 1/7: Computing base plate normal…", log_cb)
    evals, evecs = np.linalg.eigh(np.cov(points.T))
    base_normal  = evecs[:, np.argsort(evals)[0]]
    _log(f"[INFO] Base normal: {base_normal.round(3).tolist()}", log_cb)

    _log("[INFO] Step 2/7: Finding base level + adaptive threshold…", log_cb)
    proj0 = points @ base_normal
    hist, bins = np.histogram(proj0, bins=80)
    centers    = (bins[:-1] + bins[1:]) / 2
    peak_idx   = np.argmax(hist)
    base_level = centers[peak_idx]

    half_max = hist[peak_idx] / 2
    li, ri = peak_idx, peak_idx
    while li > 0 and hist[li] > half_max:             li -= 1
    while ri < len(hist) - 1 and hist[ri] > half_max: ri += 1
    slab_hw = (centers[ri] - centers[li]) / 2
    # Use 1% of the full projection range as a unit-agnostic minimum
    proj_range = proj0.max() - proj0.min()
    THRESH = max(slab_hw * 1.5, proj_range * 0.01)
    _log(f'[INFO] Base level={base_level:.4f}  slab_hw={slab_hw:.4f}  THRESH={THRESH:.4f}', log_cb)

    _log("[INFO] Step 3/7: Isolating vertical plate…", log_cb)
    above_mask = proj0 > (base_level + THRESH)
    below_mask = proj0 < (base_level - THRESH)
    above_pts  = points[above_mask] if above_mask.sum() > 10 else None
    below_pts  = points[below_mask] if below_mask.sum() > 10 else None

    if above_pts is not None and below_pts is not None:
        ta = _projected_thinness(above_pts, base_normal)
        tb = _projected_thinness(below_pts, base_normal)
        raw_vert = above_pts  if ta < tb else below_pts
        base_pts = points[~above_mask] if ta < tb else points[~below_mask]
    elif above_pts is not None:
        raw_vert, base_pts = above_pts, points[~above_mask]
    elif below_pts is not None:
        raw_vert, base_pts = below_pts, points[~below_mask]
    else:
        raise IsolationError("No points beyond base slab threshold.")

    # Contamination filter — scale tolerances with point cloud bounding box
    pp_raw = raw_vert - np.outer(raw_vert @ base_normal, base_normal)
    ev_raw, _ = np.linalg.eigh(np.cov(pp_raw.T))
    strip_width = 2 * np.sqrt(max(np.sort(ev_raw)[1], 0))
    # 'wide' threshold: 10% of the data's cross-section extent
    pp_full = points - np.outer(points @ base_normal, base_normal)
    cross_ext = np.sqrt(np.linalg.eigh(np.cov(pp_full.T))[0].max()) * 2
    wide_thresh = cross_ext * 0.10

    if strip_width > wide_thresh:
        pp_med = np.median(pp_raw, axis=0)
        dists  = np.linalg.norm(pp_raw - pp_med, axis=1)
        h_d, b_d = np.histogram(dists, bins=40)
        c_d  = (b_d[:-1] + b_d[1:]) / 2
        pk1  = np.argmax(h_d[:20])
        gap_start = pk1
        for i in range(pk1 + 1, 40):
            if h_d[i] == 0 or h_d[i] < h_d[pk1] * 0.05:
                gap_start = i; break
        gap_end = 39
        for i in range(gap_start, 40):
            if h_d[i] > h_d[pk1] * 0.05:
                gap_end = i; break
        # Clip between 10% and 40% of cross extent
        gap_thresh = np.clip((c_d[gap_start - 1] + c_d[gap_end]) / 2,
                             cross_ext * 0.10, cross_ext * 0.40)
        vert_pts   = raw_vert[dists <= gap_thresh]
        _log(f'[INFO] Contamination cleaned: {len(vert_pts)} pts (gap_thresh={gap_thresh:.4f})', log_cb)
    else:
        vert_pts = raw_vert

    if len(vert_pts) < 30:
        raise InsufficientPointsError(f"Only {len(vert_pts)} vert plate points after filtering.")

    _log(f"[INFO] Vert plate: {len(vert_pts)} pts | Base: {len(base_pts)} pts", log_cb)

    _log("[INFO] Step 4/7: Computing seam direction…", log_cb)
    vp = vert_pts - np.outer(vert_pts @ base_normal, base_normal)
    ev4, evec4 = np.linalg.eigh(np.cov(vp.T))
    seam_dir   = evec4[:, np.argmax(ev4)]
    seam_dir   = seam_dir - np.dot(seam_dir, base_normal) * base_normal
    seam_dir  /= np.linalg.norm(seam_dir)
    _log(f"[INFO] Seam direction: {seam_dir.round(3).tolist()}", log_cb)

    _log("[INFO] Step 5/7: Building cross-section frame…", log_cb)
    u_ax = np.cross(seam_dir, base_normal); u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(seam_dir, u_ax);        v_ax /= np.linalg.norm(v_ax)

    _log("[INFO] Step 6/7: Slicing along seam extent…", log_cb)
    proj_sv  = vert_pts @ seam_dir
    proj_sb  = base_pts @ seam_dir
    pmin, pmax = proj_sv.min(), proj_sv.max()
    # Use adaptive slice spacing: aim for ~50 slices regardless of units
    seam_len = pmax - pmin
    slice_step = seam_len / 50.0
    n_slices   = max(10, int(seam_len / slice_step))
    edges      = np.linspace(pmin, pmax, n_slices + 1)
    _log(f'[INFO] Seam extent: {seam_len:.4f} units → {n_slices} slices (step={slice_step:.5f})', log_cb)

    _log("[INFO] Step 7/7: Corner detection per slice…", log_cb)
    corners_a, corners_b = [], []
    for i in range(n_slices):
        lo, hi = edges[i], edges[i + 1]
        mb = (proj_sb >= lo) & (proj_sb < hi)
        mv = (proj_sv >= lo) & (proj_sv < hi)
        if mv.sum() < 5 or mb.sum() < 5:
            continue
        v_b = base_pts[mb] @ v_ax
        vh_, vb_ = np.histogram(v_b, bins=max(10, mb.sum() // 5))
        base_top = ((vb_[:-1] + vb_[1:]) / 2)[np.argmax(vh_)]
        u_v = vert_pts[mv] @ u_ax
        sp_ = (lo + hi) / 2
        corners_a.append(np.percentile(u_v, 10) * u_ax + base_top * v_ax + sp_ * seam_dir)
        corners_b.append(np.percentile(u_v, 90) * u_ax + base_top * v_ax + sp_ * seam_dir)

    if len(corners_a) < 3:
        raise InsufficientSlicesError(
            f"Only {len(corners_a)} valid slices — point cloud too sparse.")

    corners_a = np.array(corners_a)
    corners_b = np.array(corners_b)

    anchor_a, dir_a, start_a, end_a = _fit_line_pca(corners_a)
    anchor_b, dir_b, start_b, end_b = _fit_line_pca(corners_b)

    waypoints_a = _make_waypoints(start_a, end_a)
    waypoints_b = _make_waypoints(start_b, end_b)

    timing_ms = (time.perf_counter() - t_start) * 1000.0
    _log(f"[INFO] Valid slices: {len(corners_a)} | Timing: {timing_ms:.1f}ms", log_cb)

    return {
        'seam_dir':    _normalize(dir_a + dir_b * np.sign(np.dot(dir_a, dir_b))),
        'seam_a':      anchor_a, 'seam_b': anchor_b,
        'line_a':      {'anchor': anchor_a, 'direction': dir_a, 'start': start_a, 'end': end_a},
        'line_b':      {'anchor': anchor_b, 'direction': dir_b, 'start': start_b, 'end': end_b},
        'waypoints_a': waypoints_a, 'waypoints_b': waypoints_b,
        'corners_a':   corners_a,   'corners_b':   corners_b,
        'n_slices':    len(corners_a),
        'base_normal': base_normal,
        'base_level':  float(base_level),
        'timing_ms':   timing_ms,
    }


# ── Public entrypoint — matches seam_detector.run_seam_detection() signature ──

def run_seam_detection(log_cb=None):
    """
    AdaptiveSlice seam detection. Reads merged_xyzrgb.npy, writes
    seam_results.json and seams.json using the same schema as Triplane RANSAC.
    """
    _log("[STEP] SEAM_DETECT_START", log_cb)

    results_path = os.path.join(DATA_DIR, "seam_results.json")
    if os.path.exists(results_path):
        os.remove(results_path)

    pcd_file = os.path.join(DATA_DIR, "merged_xyzrgb.npy")
    if not os.path.exists(pcd_file):
        _log("[ERROR] Missing merged_xyzrgb.npy. Run Process step first.", log_cb)
        return

    _log("[INFO] Loading point cloud…", log_cb)
    data = np.load(pcd_file)
    points = data[:, :3]
    _log(f"[INFO] {len(points):,} points loaded.", log_cb)

    try:
        raw = _detect(points, log_cb=log_cb)
    except (IsolationError, InsufficientPointsError, InsufficientSlicesError) as e:
        _log(f"[ERROR] {e.__class__.__name__} [{e.ERROR_CODE}]: {e}", log_cb)
        with open(results_path, "w") as f:
            json.dump({"error": str(e), "error_code": e.ERROR_CODE}, f, indent=2)
        return
    except Exception as e:
        _log(f"[ERROR] Unexpected failure: {e}", log_cb)
        with open(results_path, "w") as f:
            json.dump({"error": str(e)}, f, indent=2)
        return

    # ── Normalise output to seam_detector.py schema ──────────────────────────
    la, lb = raw['line_a'], raw['line_b']

    # Compute a side vector (offset direction for tool offsets)
    side_mm = 3.0
    side_a  = _normalize(np.cross(raw['base_normal'], la['direction']))
    side_b  = _normalize(np.cross(raw['base_normal'], lb['direction']))

    res = {
        "seam1": {
            "start":       la['start'].tolist(),
            "end":         la['end'].tolist(),
            "start_left":  (la['start'] + side_a * side_mm).tolist(),
            "end_left":    (la['end']   + side_a * side_mm).tolist(),
            "start_right": (la['start'] - side_a * side_mm).tolist(),
            "end_right":   (la['end']   - side_a * side_mm).tolist(),
            "travel_dir":  la['direction'].tolist(),
            "side_dir":    side_a.tolist(),
            "bisector":    _normalize(raw['base_normal'] + la['direction']).tolist(),
        },
        "seam2": {
            "start":       lb['start'].tolist(),
            "end":         lb['end'].tolist(),
            "start_left":  (lb['start'] + side_b * side_mm).tolist(),
            "end_left":    (lb['end']   + side_b * side_mm).tolist(),
            "start_right": (lb['start'] - side_b * side_mm).tolist(),
            "end_right":   (lb['end']   - side_b * side_mm).tolist(),
            "travel_dir":  lb['direction'].tolist(),
            "side_dir":    side_b.tolist(),
            "bisector":    _normalize(raw['base_normal'] + lb['direction']).tolist(),
        },
        "algorithm_info": {
            "name":       "AdaptiveSlice (SEAM-LOC 009)",
            "n_slices":   raw['n_slices'],
            "timing_ms":  raw['timing_ms'],
            "base_normal": raw['base_normal'].tolist(),
            "base_level":  raw['base_level'],
        },
        # No "planes" key since AdaptiveSlice doesn't produce RANSAC planes
    }

    with open(results_path, "w") as f:
        json.dump(res, f, indent=4)

    legacy = {
        "seam1": {"start": la['start'].tolist(), "end": la['end'].tolist()},
        "seam2": {"start": lb['start'].tolist(), "end": lb['end'].tolist()},
    }
    with open(os.path.join(DATA_DIR, "seams.json"), "w") as f:
        json.dump(legacy, f, indent=4)

    _log(f"[STEP] SEAM_COMPUTED count=2", log_cb)
    _log(f"[INFO] Seam 1: {la['start'].round(3).tolist()} → {la['end'].round(3).tolist()}", log_cb)
    _log(f"[INFO] Seam 2: {lb['start'].round(3).tolist()} → {lb['end'].round(3).tolist()}", log_cb)
    _log("[STEP] SEAM_DETECT_COMPLETE", log_cb)


if __name__ == "__main__":
    run_seam_detection()
