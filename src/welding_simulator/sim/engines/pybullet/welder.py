"""
sim_weld.py — Stage 4: Welding Execution (PyBullet Backend)
Starts PyBullet headless, rebuilds scene, then moves the camera/robot along the
detected weld seam paths loaded from data/latest/seams.json to generate a video.
"""
import sys, os
import pybullet as p
import pybullet_data
import time
import json
import numpy as np
import cv2
from pathlib import Path
import subprocess

ROOT = str(Path(__file__).resolve().parents[5])
DATA_DIR = os.path.join(ROOT, "data", "latest")

def _log(msg, log_cb):
    if log_cb:
        log_cb(msg)
    else:
        print(msg, flush=True)

def get_or_create_engine():
    if not p.isConnected():
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    else:
        p.resetSimulation()
    p.setGravity(0, 0, -9.81)

def create_table(dimensions, position=(0,0,0)):
    planeId = p.loadURDF("plane.urdf")
    table_half_extents = [dimensions[0]/2, dimensions[1]/2, dimensions[2]/2]
    table_pos_z_center = [position[0], position[1], position[2] + table_half_extents[2]]
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.85, 0.85, 1.0, 1])
    tableId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visBoxId, basePosition=table_pos_z_center)
    return tableId

def build_pybullet_joint(cfg, position):
    color = [0.6, 0.6, 0.6, 1.0]
    if cfg.get("joint_type") == "tee":
        bw = cfg.get("bw", 0.15)
        bl = cfg.get("bl", 0.15)
        bt = cfg.get("bt", 0.025)
        sh = cfg.get("sh", 0.15)
        st = cfg.get("st", 0.025)
        
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bw/2, bl/2, bt/2])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[bw/2, bl/2, bt/2], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + bt/2])
        
        p2_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bw/2, st/2, sh/2])
        v2_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[bw/2, st/2, sh/2], rgbaColor=color)
        b2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p2_id, baseVisualShapeIndex=v2_id, basePosition=[position[0], position[1], position[2] + bt + sh/2])
    else:
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.05])

def run_weld(cfg=None, seams=None, log_cb=None):
    if seams is None:
        try:
            with open(os.path.join(DATA_DIR, "seams.json")) as f:
                seams = json.load(f)
        except FileNotFoundError:
            _log("[ERROR] No seams.json found. Please run the Process step first.", log_cb)
            return

    if cfg is None:
        try:
            with open(os.path.join(DATA_DIR, "config.json")) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = {"joint_type": "tee"}

    seg1_start = np.array(seams.get("seam1", {}).get("start", [0, 0, 0]))
    seg1_end   = np.array(seams.get("seam1", {}).get("end", [0, 0, 0]))
    seg2_start = np.array(seams.get("seam2", {}).get("start", [0, 0, 0]))
    seg2_end   = np.array(seams.get("seam2", {}).get("end", [0, 0, 0]))
    _log("[STEP] SEAMS_LOADED", log_cb)

    get_or_create_engine()

    table_dims = (1.5, 3, 1)
    table_pos  = (0.75, 0, 0)
    create_table(table_dims, table_pos)
    build_pybullet_joint(cfg, position=(0.75, 0, table_dims[2]))

    robotStartPos = [0.2, 0, table_dims[2]]
    robotStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
    try:
        robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)
    except Exception:
        pass

    FPS = 2
    TRANSITION_SECONDS = 1
    WELD_SECONDS = 2
    home = np.array([0.5, -0.50, 1.5])

    waypoints = []

    def add_trajectory(start, end, steps):
        for i in range(int(steps)):
            a = i / (steps - 1) if steps > 1 else 1.0
            waypoints.append(start + a * (end - start))

    add_trajectory(home, seg1_start, FPS * TRANSITION_SECONDS)
    add_trajectory(seg1_start, seg1_end, FPS * WELD_SECONDS)
    add_trajectory(seg1_end, seg2_start, FPS * TRANSITION_SECONDS)
    add_trajectory(seg2_start, seg2_end, FPS * WELD_SECONDS)
    add_trajectory(seg2_end, home, FPS * TRANSITION_SECONDS)

    waypoints = np.array(waypoints)

    _log(f"[STEP] WELD_START total={len(waypoints)}", log_cb)

    video_frames_dir = os.path.join(DATA_DIR, "video_frames")
    os.makedirs(video_frames_dir, exist_ok=True)
    frame_idx = 0

    width, height = 800, 600
    fov, near, far = 60, 0.1, 1.5
    proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

    for i, pos in enumerate(waypoints):
        cam_eye = pos + np.array([-0.2, -0.2, 0.2])
        view_matrix = p.computeViewMatrix(cameraEyePosition=cam_eye,
                                          cameraTargetPosition=pos,
                                          cameraUpVector=[0, 0, 1])

        light_dir = [pos[0] - cam_eye[0], pos[1] - cam_eye[1], 2.0]
        _, _, rgbImg, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, shadow=0, lightDirection=light_dir, lightColor=[1, 1, 1])
        
        rgb_arr = np.reshape(rgbImg, (height, width, 4))[:,:,:3]
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(video_frames_dir, f"weld_{frame_idx:05d}.jpg"), bgr)
        frame_idx += 1

        p.stepSimulation()
        _log(f"[STEP] WELD_WAYPOINT_DONE index={i}", log_cb)

    _log("[STEP] ENCODING_VIDEO", log_cb)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-nostdin", "-framerate", "5", "-i", os.path.join(video_frames_dir, "weld_%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", os.path.join(DATA_DIR, "weld_video.mp4")
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except Exception as e:
        _log(f"    [ERROR] ffmpeg encoding failed: {e}", log_cb)

    _log("[STEP] WELD_COMPLETE", log_cb)

if __name__ == "__main__":
    run_weld()
