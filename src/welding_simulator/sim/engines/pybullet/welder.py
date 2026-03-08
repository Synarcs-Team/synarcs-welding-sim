"""
sim_weld.py — Stage 4: Welding Execution (PyBullet Backend)
Starts PyBullet headless, rebuilds scene, then moves the camera/robot along the
detected weld seam paths loaded from data/latest/seams.json to generate a video.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

import pybullet as p
import pybullet_data
import time
import json
import numpy as np
import cv2

# Connect to PyBullet headless
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

from pathlib import Path
ROOT     = str(Path(__file__).resolve().parents[5])
DATA_DIR = os.path.join(ROOT, "data", "latest")

# ── Load seams and config ─────────────────────────────────────────────────────
try:
    with open(os.path.join(DATA_DIR, "seams.json")) as f:
        seams = json.load(f)
except FileNotFoundError:
    print("[ERROR] No seams.json found. Please run the Process step first.")
    sys.exit(1)

try:
    with open(os.path.join(DATA_DIR, "config.json")) as f:
        cfg = json.load(f)
except FileNotFoundError:
    cfg = {"joint_type": "tee"}

seg1_start = np.array(seams.get("seam1", {}).get("start", [0, 0, 0]))
seg1_end   = np.array(seams.get("seam1", {}).get("end", [0, 0, 0]))
seg2_start = np.array(seams.get("seam2", {}).get("start", [0, 0, 0]))
seg2_end   = np.array(seams.get("seam2", {}).get("end", [0, 0, 0]))
print("[STEP] SEAMS_LOADED", flush=True)

# ── Build Scene ────────────────────────────────────────────────────────────────
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
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.1, 0.01])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.1, 0.01], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.01])
        
        p2_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.01, 0.1])
        v2_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.01, 0.1], rgbaColor=color)
        b2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p2_id, baseVisualShapeIndex=v2_id, basePosition=[position[0], position[1], position[2] + 0.1 + 0.02])
    else:
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.05])

table_dims = (1.5, 3, 1)
table_pos  = (0, -1.5, 0)
create_table(table_dims, table_pos)
build_pybullet_joint(cfg, position=(0.75, 0, table_dims[2]))

# Add dummy robot
robotStartPos = [0.2, 0, table_dims[2]]
robotStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
try:
    robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)
except Exception:
    pass

# ── Build waypoint list ───────────────────────────────────────────────────────
STEPS=10
home=np.array([0.5, -0.50, 1.5])
waypoints=[home]
# Move to seam 1
for i in range(STEPS):
    a=i/(STEPS-1)
    waypoints.append(seg1_start+a*(seg1_end-seg1_start))
# Move to seam 2
for i in range(STEPS):
    a=i/(STEPS-1)
    waypoints.append(seg2_start+a*(seg2_end-seg2_start))
waypoints.append(home)
waypoints=np.array(waypoints)

print(f"[STEP] WELD_START total={len(waypoints)}", flush=True)

# Prepare for video recording
video_frames_dir = os.path.join(DATA_DIR, "video_frames")
os.makedirs(video_frames_dir, exist_ok=True)
frame_idx = 0

width, height = 800, 600
fov, near, far = 60, 0.1, 1.5
proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

for i, pos in enumerate(waypoints):
    # Make the camera follow the "torch" hovering slightly above the weld point
    cam_eye = pos + np.array([-0.2, -0.2, 0.2])
    view_matrix = p.computeViewMatrix(cameraEyePosition=cam_eye,
                                      cameraTargetPosition=pos,
                                      cameraUpVector=[0, 0, 1])

    _, _, rgbImg, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    
    # Simple video frame capture
    rgb_arr = np.reshape(rgbImg, (height, width, 4))[:,:,:3]
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(video_frames_dir, f"weld_{frame_idx:05d}.jpg"), bgr)
    frame_idx += 1

    p.stepSimulation()
    print(f"[STEP] WELD_WAYPOINT_DONE index={i}", flush=True)

# Encode video
import subprocess
print("[STEP] ENCODING_VIDEO", flush=True)
try:
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-framerate", "30", "-i", os.path.join(video_frames_dir, "weld_%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(DATA_DIR, "weld_video.mp4")
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
except Exception as e:
    print(f"    [ERROR] ffmpeg encoding failed: {e}", flush=True)

print("[STEP] WELD_COMPLETE", flush=True)
p.disconnect()
os._exit(0)
