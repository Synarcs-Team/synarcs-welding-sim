"""
seam_detector.py — Stage 4: Seam Detection (T-Joint only)
Reads merged_xyzrgb.npy, fits planes using RANSAC, and calculates toolpaths.
Adapted from t_joint_two_seams_deterministic.py
"""
import sys, os, json, math, time
import numpy as np

# We'll import Open3D for PCD handling and plane fitting internals if needed, 
# but the custom RANSAC relies on numpy.
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
ROOT     = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(ROOT, "data", "latest")

# --- Constants & Parameters ---
SEED = 0
RANSAC_ITERS = 10000
MAX_DIST_MM = 0.005
MAX_ANG_DEG = 20.0
PLANE_TOL1_MM = 0.01
PLANE_TOL2_MM = 0.01
TUBE_RADIUS_MM = 0.02
SIDE_OFFSET_MM = 0.003
TRIM_PCT = (2.0, 98.0)
CONSTRAIN_BASE_TO_UP = True
WORLD_UP = np.array([0.0, 0.0, 1.0])
BASE_MAX_ANG_TO_UP_DEG = 25.0

def _log(msg, log_cb=None):
    if log_cb: log_cb(msg)
    else: print(msg, flush=True)

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * 0.0 if n < eps else v / n

def point_plane_distance(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return np.abs(P @ n + d)

def plane_from_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, eps: float = 1e-9):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    if np.linalg.norm(n) < eps: return None, None
    n = normalize(n)
    d = -float(np.dot(n, p1))
    return n, d

def tls_plane_fit(P: np.ndarray):
    C = P.mean(axis=0)
    A = P - C
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    n = normalize(vh[-1, :])
    d = -float(np.dot(n, C))
    return n, d

class PlaneModel:
    def __init__(self, n, d, inliers):
        self.n = n
        self.d = d
        self.inliers = inliers

def ransac_plane_deterministic(P, max_dist, num_iters, seed, perpendicular_to=None, max_ang_deg=20.0, near_normal=None, min_inliers=200):
    rng = np.random.default_rng(seed)
    N = len(P)
    if N < 3: return None

    best_count = -1
    best_inliers = None
    best_n = None
    best_d = None

    perp_limit = math.sin(math.radians(max_ang_deg)) if perpendicular_to is not None else None
    n_ref_perp = normalize(perpendicular_to) if perpendicular_to is not None else None

    near_limit = math.cos(math.radians(max_ang_deg)) if near_normal is not None else None
    n_ref_near = normalize(near_normal) if near_normal is not None else None

    idx = np.arange(N)
    for _ in range(num_iters):
        sample = rng.choice(idx, size=3, replace=False)
        n, d = plane_from_3pts(P[sample[0]], P[sample[1]], P[sample[2]])
        if n is None: continue

        if near_limit is not None and abs(float(np.dot(n, n_ref_near))) < near_limit: continue
        if perp_limit is not None and abs(float(np.dot(n, n_ref_perp))) > perp_limit: continue

        dist = point_plane_distance(P, n, d)
        inl = np.where(dist <= max_dist)[0]
        c = inl.size
        if c > best_count:
            best_count = c; best_inliers = inl; best_n = n; best_d = d

    if best_inliers is None or best_count < min_inliers: return None
    n_refined, d_refined = tls_plane_fit(P[best_inliers])
    if float(np.dot(n_refined, best_n)) < 0:
        n_refined = -n_refined; d_refined = -d_refined
    return PlaneModel(n_refined, float(d_refined), best_inliers)

def intersect_two_planes(n1, d1, n2, d2):
    v = np.cross(n1, n2)
    nv = np.linalg.norm(v)
    if nv < 1e-12: return None, None
    v = v / nv
    M = np.vstack([n1, n2, v])
    b = -np.array([d1, d2, 0.0], dtype=float)
    try:
        p0 = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        p0 = np.linalg.lstsq(M, b, rcond=None)[0]
    return p0, v

def point_line_distance(P, p0, v_unit):
    return np.linalg.norm(np.cross(P - p0, v_unit[None, :]), axis=1)

def seam_segment_from_planes(P, n_base, d_base, n_face, d_face):
    p0, v = intersect_two_planes(n_base, d_base, n_face, d_face)
    if p0 is None: raise RuntimeError("Cannot intersect planes (nearly parallel).")
    
    v_ref = normalize(np.cross(n_base, n_face))
    if float(np.dot(v, v_ref)) < 0: v = -v

    d1 = point_plane_distance(P, n_base, d_base)
    d2 = point_plane_distance(P, n_face, d_face)
    Q = P[(d1 <= PLANE_TOL1_MM) & (d2 <= PLANE_TOL2_MM)]
    if len(Q) == 0: raise RuntimeError("No points near both planes for this seam.")

    dl = point_line_distance(Q, p0, v)
    Q2 = Q[dl <= TUBE_RADIUS_MM]
    if len(Q2) == 0: raise RuntimeError("No points within tubeRadius for this seam.")

    t = (Q2 - p0) @ v
    tMin = float(np.percentile(t, TRIM_PCT[0]))
    tMax = float(np.percentile(t, TRIM_PCT[1]))
    if not (np.isfinite(tMin) and np.isfinite(tMax) and (tMax > tMin)):
        raise RuntimeError("Degenerate t-range for this seam.")

    return p0 + tMin * v, p0 + tMax * v, v

def toolpath_offsets(A, B, v, n_base, offset):
    side = normalize(np.cross(n_base, v))
    return A + offset * side, B + offset * side, A - offset * side, B - offset * side, side

def run_seam_detection(log_cb=None):
    _log("[STEP] SEAM_DETECT_START", log_cb)
    
    # Clear previous results to prevent stale data on failure
    results_path = os.path.join(DATA_DIR, "seam_results.json")
    if os.path.exists(results_path):
        os.remove(results_path)
    
    try:
        with open(os.path.join(DATA_DIR, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("joint_type", "tee") != "tee":
            _log("[WARN] Seam detection is currently only supported for Tee joints.", log_cb)
            _log("[STEP] SEAM_DETECT_COMPLETE", log_cb)
            return
    except Exception:
        pass # missing config.json is fine, default to thinking it is a tee joint
        
    pcd_file = os.path.join(DATA_DIR, "merged_xyzrgb.npy")
    if not os.path.exists(pcd_file):
        _log("[ERROR] Missing merged_xyzrgb.npy. Run Process step first.", log_cb)
        return

    _log("[INFO] Loading point cloud...", log_cb)
    data = np.load(pcd_file)
    P = data[:, :3]
    
    # 1. Base Plane
    _log("[INFO] Fitting Base Plane (RANSAC)...", log_cb)
    base_model = ransac_plane_deterministic(
        P, max_dist=MAX_DIST_MM, num_iters=RANSAC_ITERS, seed=SEED,
        near_normal=WORLD_UP if CONSTRAIN_BASE_TO_UP else None,
        max_ang_deg=BASE_MAX_ANG_TO_UP_DEG, min_inliers=200
    )
    if not base_model:
        _log("[ERROR] Failed to fit base plane.", log_cb)
        return
    n1, d1, in1 = base_model.n, base_model.d, base_model.inliers
    _log(f"[STEP] PLANE_FIT_BASE inliers={len(in1)} normal={np.round(n1, 3).tolist()}", log_cb)
    
    mask1 = np.ones(len(P), dtype=bool); mask1[in1] = False
    P_rem1 = P[mask1]

    # 2. Stem Face 1
    _log("[INFO] Fitting Stem Face 1...", log_cb)
    stem1_model = ransac_plane_deterministic(
        P_rem1, max_dist=MAX_DIST_MM, num_iters=RANSAC_ITERS, seed=SEED+1,
        perpendicular_to=n1, max_ang_deg=MAX_ANG_DEG, min_inliers=100
    )
    if not stem1_model:
        _log("[ERROR] Failed to fit stem face #1.", log_cb)
        return
    n2, d2, in2_local = stem1_model.n, stem1_model.d, stem1_model.inliers
    in2 = np.where(mask1)[0][in2_local]
    _log(f"[STEP] PLANE_FIT_STEM1 inliers={len(in2)} normal={np.round(n2, 3).tolist()}", log_cb)

    mask2 = mask1.copy(); mask2[in2] = False
    P_rem2 = P[mask2]

    # 3. Stem Face 2
    _log("[INFO] Fitting Stem Face 2...", log_cb)
    stem2_model = ransac_plane_deterministic(
        P_rem2, max_dist=MAX_DIST_MM, num_iters=RANSAC_ITERS, seed=SEED+2,
        perpendicular_to=n1, max_ang_deg=MAX_ANG_DEG, min_inliers=100
    )
    if not stem2_model:
        _log("[ERROR] Failed to fit stem face #2.", log_cb)
        return
    n3, d3, in3_local = stem2_model.n, stem2_model.d, stem2_model.inliers
    in3 = np.where(mask2)[0][in3_local]
    _log(f"[STEP] PLANE_FIT_STEM2 inliers={len(in3)} normal={np.round(n3, 3).tolist()}", log_cb)

    # 4. Compute Seams
    _log("[INFO] Computing Seam Intersections and Trimming...", log_cb)
    try:
        A1, B1, v1 = seam_segment_from_planes(P, n1, d1, n2, d2)
        A2, B2, v2 = seam_segment_from_planes(P, n1, d1, n3, d3)
    except Exception as e:
        _log(f"[ERROR] Seam computation failed: {e}", log_cb)
        print(f"CRITICAL SEAM ERROR: {e}", file=sys.stderr, flush=True)
        error_res = {"error": f"Seam mathematical computation failed: {e}"}
        with open(os.path.join(DATA_DIR, "seam_results.json"), "w") as f:
            json.dump(error_res, f, indent=4)
        return

    A1L, B1L, A1R, B1R, side1 = toolpath_offsets(A1, B1, v1, n1, SIDE_OFFSET_MM)
    A2L, B2L, A2R, B2R, side2 = toolpath_offsets(A2, B2, v2, n1, SIDE_OFFSET_MM)

    bis1 = normalize(n1 + n2) if np.linalg.norm(n1 + n2) > 0 else n1
    bis2 = normalize(n1 + n3) if np.linalg.norm(n1 + n3) > 0 else n1
    
    _log("[STEP] SEAM_COMPUTED count=2", log_cb)

    # 5. Save Results
    res = {
        "seam1": {
            "start": A1.tolist(), "end": B1.tolist(),
            "start_left": A1L.tolist(), "end_left": B1L.tolist(),
            "start_right": A1R.tolist(), "end_right": B1R.tolist(),
            "travel_dir": v1.tolist(), "side_dir": side1.tolist(), "bisector": bis1.tolist()
        },
        "seam2": {
            "start": A2.tolist(), "end": B2.tolist(),
            "start_left": A2L.tolist(), "end_left": B2L.tolist(),
            "start_right": A2R.tolist(), "end_right": B2R.tolist(),
            "travel_dir": v2.tolist(), "side_dir": side2.tolist(), "bisector": bis2.tolist()
        },
        "planes": {
            "base": {"normal": n1.tolist(), "d": d1, "inlier_count": len(in1)},
            "stem1": {"normal": n2.tolist(), "d": d2, "inlier_count": len(in2)},
            "stem2": {"normal": n3.tolist(), "d": d3, "inlier_count": len(in3)},
        },
        "params": {"seed": SEED, "ransac_iters": RANSAC_ITERS, "max_dist": MAX_DIST_MM}
    }

    with open(os.path.join(DATA_DIR, "seam_results.json"), "w") as f:
        json.dump(res, f, indent=4)
        
    # Optional: overwrite the legacy seams.json for backward compatibility if needed by other components
    # Although the new Step 4 will likely fetch seam_results.json
    legacy_seams = {
        "seam1": {"start": A1.tolist(), "end": B1.tolist()},
        "seam2": {"start": A2.tolist(), "end": B2.tolist()}
    }
    with open(os.path.join(DATA_DIR, "seams.json"), "w") as f:
        json.dump(legacy_seams, f, indent=4)

    _log("[STEP] SEAM_DETECT_COMPLETE", log_cb)

if __name__ == "__main__":
    run_seam_detection()
