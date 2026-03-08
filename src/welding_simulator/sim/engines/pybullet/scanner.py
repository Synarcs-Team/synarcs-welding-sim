"""
sim_scan.py — Stage 1: Scanning (PyBullet Backend)
Starts PyBullet headless, builds scene, calculates IK to 5 camera positions,
captures point clouds + RGB images, saves them to data/latest/.
Prints progress markers for the web app to parse.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)   # flush every line for streaming

import pybullet as p
import pybullet_data
import time
import json
import numpy as np
from PIL import Image
import open3d as o3d
import cv2

# Connect to PyBullet (DIRECT = headless, GUI = visual)
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# ── Paths ─────────────────────────────────────────────────────────────────────
from pathlib import Path
ROOT        = str(Path(__file__).resolve().parents[5])
CONFIG_PATH = os.path.join(ROOT, "config", "joint_config.json")
DATA_DIR    = os.path.join(ROOT, "data", "latest")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────
try:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
except FileNotFoundError:
    cfg = {"joint_type": "tee"}
print(f"[STEP] CONFIG_LOADED", flush=True)

# ── Helper functions ──────────────────────────────────────────────────────────

def create_table(dimensions, position=(0,0,0)):
    # Create static floor plane
    planeId = p.loadURDF("plane.urdf")
    
    # Create table collision and visual bounding box
    table_half_extents = [dimensions[0]/2, dimensions[1]/2, dimensions[2]/2]
    table_pos_z_center = [position[0], position[1], position[2] + table_half_extents[2]]
    
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.85, 0.85, 1.0, 1])
    tableId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visBoxId, basePosition=table_pos_z_center)
    return tableId

# Need simple representations for joint pieces in PyBullet
def build_pybullet_joint(cfg, position):
    """
    Since PyBullet doesn't use standard Open3D meshes natively without saving to disk first, 
    we will simulate the joint shape using simple box primitives just for visual/camera rendering.
    """
    # Use the joint factory logic to get overall bounds, but we must manually draw boxes
    # This acts as a proxy object for the cameras to capture. 
    # For a true simulation we would write the Open3D mesh to an OBJ and load it.
    
    # A generic proxy box based on joint type for now to ensure rendering works.
    color = [0.6, 0.6, 0.6, 1.0] # Metallic grey
    
    if cfg.get("joint_type") == "tee":
        # Base plate
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.1, 0.01])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.1, 0.01], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.01])
        
        # Vertical plate
        p2_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.01, 0.1])
        v2_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.01, 0.1], rgbaColor=color)
        b2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p2_id, baseVisualShapeIndex=v2_id, basePosition=[position[0], position[1], position[2] + 0.1 + 0.02])
        return [b1, b2], (0.5, 0.2, 0.22)
    else:
        # Generic block for other joints as a fallback
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.05])
        return [b1], (0.4, 0.3, 0.1)

# ── Build world ───────────────────────────────────────────────────────────────

table_dims = (1.5, 3, 1)
table_pos  = (0, -1.5, 0)
tableId = create_table(table_dims, table_pos)

joint_parts, joint_bbox = build_pybullet_joint(cfg, position=(0.75, 0, table_dims[2]))

# UR10
robotStartPos = [0.2, 0, table_dims[2]]
robotStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
# PyBullet standard library doesn't contain UR10 by default, using Kuka for dummy visualization if missing, 
# but assuming you have a urdf or it fetches a generic arm
try:
    robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)
    numJoints = p.getNumJoints(robotId)
    ee_link_idx = numJoints - 1
except Exception as e:
    print(f"[WARN] Failed to load arm urdf: {e}. Camera will move freely.")
    robotId = None
    ee_link_idx = -1

# ── Scan loop ─────────────────────────────────────────────────────────────────
bx, by, bz = joint_bbox
cx, cy = 0.75, 0.0                      
z_scan = table_dims[2] + bz + 0.35     
y_offset = float(np.clip(by / 2.0 + 0.25, 0.25, 0.50))

scan_positions = np.array([
    [cx,      cy,          z_scan + 0.15],  # Top-down
    [cx + 0.35, cy + y_offset, z_scan],     # Front-right
    [cx + 0.35, cy - y_offset, z_scan],     # Front-left
    [cx - 0.25, cy + y_offset, z_scan],     # Slight back-right
    [cx - 0.25, cy - y_offset, z_scan],     # Slight back-left
])
table_target = np.array([cx, cy, table_dims[2] + bz / 2.0])

# Prepare for video recording
video_frames_dir = os.path.join(DATA_DIR, "video_frames")
os.makedirs(video_frames_dir, exist_ok=True)
frame_idx = 0

def convert_depth_to_pointcloud(depth, rgb, view_matrix, proj_matrix, width, height, far=1.5, near=0.1):
    # PyBullet returns depth buffer spanning 0-1, need to convert to real z
    depth_buffer = np.reshape(depth, [height, width])
    depthImg = far * near / (far - (far - near) * depth_buffer)
    
    # Calculate point cloud
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order='F')
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order='F')
    inverse_view = np.linalg.inv(view_matrix)
    inverse_proj = np.linalg.inv(proj_matrix)
    
    points = []
    colors = []
    
    rgb = np.reshape(rgb, [height, width, 4])[:,:,:3] # drop alpha D
    
    for y in range(height):
        for x in range(width):
            z = depth_buffer[y, x]
            if z >= 1.0: # Background
                continue
            
            # NDC
            ndc_x = (2.0 * x - width) / width
            ndc_y = -(2.0 * y - height) / height # PyBullet Y is inverted
            
            ndc_pos = np.array([ndc_x, ndc_y, 2.0 * z - 1.0, 1.0])
            clip_pos = inverse_proj @ ndc_pos
            view_pos = clip_pos / clip_pos[3]
            world_pos = inverse_view @ view_pos
            
            points.append(world_pos[:3])
            colors.append(rgb[y,x] / 255.0)
            
    return np.array(points), np.array(colors)

width, height = 800, 600
fov, near, far = 60, 0.1, 1.5
proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

point_clouds, rgbs, cam_positions, cam_orientations = [], [], [], []
print(f"[STEP] SCAN_START total={len(scan_positions)}", flush=True)

for i, pos in enumerate(scan_positions):
    # Step simulation to settle
    for _ in range(10): p.stepSimulation()
    
    view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                      cameraTargetPosition=table_target,
                                      cameraUpVector=[0, 0, 1])

    # Render image
    _, _, rgbImg, depthImg, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    
    # Simple video frame capture
    rgb_arr = np.reshape(rgbImg, (height, width, 4))[:,:,:3]
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(video_frames_dir, f"{frame_idx:05d}.jpg"), bgr)
    frame_idx += 1
    
    pts, clrs = convert_depth_to_pointcloud(depthImg, rgbImg, view_matrix, proj_matrix, width, height, far, near)

    point_clouds.append((pts, clrs))
    rgbs.append(rgb_arr)
    cam_positions.append(pos)
    
    # Fake orientation matrix from look-at for compatibility
    cam_orientations.append([0,0,0,1]) # Placeholder orientation
    print(f"[STEP] SCAN_POSITION_DONE index={i}", flush=True)

# ── Save outputs ──────────────────────────────────────────────────────────────
print("[STEP] SAVING_DATA", flush=True)
for i, ((pts, clrs), rgb, cam_pos, cam_ori) in enumerate(zip(point_clouds, rgbs, cam_positions, cam_orientations)):
    if len(pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(clrs)
        o3d.io.write_point_cloud(os.path.join(DATA_DIR, f"scan_{i}.pcd"), pcd)
    else:
        # Fallback empty cloud
        o3d.io.write_point_cloud(os.path.join(DATA_DIR, f"scan_{i}.pcd"), o3d.geometry.PointCloud())
        
    np.save(os.path.join(DATA_DIR, f"cam_pos_{i}.npy"), cam_pos)
    np.save(os.path.join(DATA_DIR, f"cam_ori_{i}.npy"), cam_ori)
    img = Image.fromarray(rgb)
    img.save(os.path.join(DATA_DIR, f"rgb_{i}.jpg"))

# Encode video from captured frames using ffmpeg
import subprocess
print("[STEP] ENCODING_VIDEO", flush=True)
try:
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-framerate", "30", "-i", os.path.join(video_frames_dir, "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(DATA_DIR, "scan_video.mp4")
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
except Exception as e:
    print(f"    [ERROR] ffmpeg encoding failed: {e}", flush=True)

# Save config used for this session
with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
    json.dump(cfg, f, indent=4)

print(f"[STEP] SCAN_COMPLETE scans={len(point_clouds)}", flush=True)
p.disconnect()
os._exit(0)
