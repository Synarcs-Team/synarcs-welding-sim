"""
sim_scan.py — Stage 1: Scanning (PyBullet Backend)
Starts PyBullet headless, builds scene, calculates IK to 5 camera positions,
captures point clouds + RGB images, saves them to data/latest/.
Streams logs via callback.
"""
import sys, os
import pybullet as p
import pybullet_data
import time
import json
import numpy as np
from PIL import Image
import open3d as o3d
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
        return [b1, b2], (bw, bl, bt + sh)
    else:
        p1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05])
        v1_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.05], rgbaColor=color)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p1_id, baseVisualShapeIndex=v1_id, basePosition=[position[0], position[1], position[2] + 0.05])
        return [b1], (0.4, 0.3, 0.1)

def convert_depth_to_pointcloud(depth, rgb, view_matrix, proj_matrix, width, height, far=2.0, near=0.01):
    depth_buffer = np.reshape(depth, [height, width])
    valid_mask = depth_buffer < 0.99
    if not np.any(valid_mask):
        return np.array([]), np.array([])
        
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x[valid_mask]
    y = y[valid_mask]
    z = depth_buffer[valid_mask]
    
    ndc_x = (2.0 * x - width) / width
    ndc_y = -(2.0 * y - height) / height
    ndc_z = 2.0 * z - 1.0
    ndc_pos = np.stack([ndc_x, ndc_y, ndc_z, np.ones_like(ndc_z)], axis=0)
    
    inv_proj = np.linalg.inv(np.asarray(proj_matrix).reshape([4, 4], order='F'))
    inv_view = np.linalg.inv(np.asarray(view_matrix).reshape([4, 4], order='F'))
    
    clip_pos = inv_proj @ ndc_pos
    view_pos = clip_pos / clip_pos[3, :]
    world_pos = inv_view @ view_pos
    
    points = world_pos[:3, :].T
    rgb_buffer = np.reshape(rgb, [height, width, 4])[:,:,:3]
    colors = rgb_buffer[valid_mask] / 255.0
    
    return points, colors

def run_scan(cfg=None, log_cb=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if cfg is None:
        config_path = os.path.join(ROOT, "config", "joint_config.json")
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = {"joint_type": "tee"}
            
    _log("[STEP] CONFIG_LOADED", log_cb)
    get_or_create_engine()
    
    table_dims = (1.5, 3, 1)
    table_pos  = (0.75, 0, 0)
    tableId = create_table(table_dims, table_pos)
    
    joint_parts, joint_bbox = build_pybullet_joint(cfg, position=(0.75, 0, table_dims[2]))
    
    robotStartPos = [0.2, 0, table_dims[2]]
    robotStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
    try:
        robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)
    except Exception as e:
        _log(f"[WARN] Failed to load arm urdf: {e}. Camera will move freely.", log_cb)
        robotId = None

    bx, by, bz = joint_bbox
    cx, cy = 0.75, 0.0

    # Calculate key heights from the actual joint geometry
    joint_top_z  = table_dims[2] + bz          # top of stem
    joint_mid_z  = table_dims[2] + bz / 2.0    # vertical center of joint
    joint_base_z = table_dims[2]               # table/base interface

    # Lateral distance: far enough to see the full joint width
    side_dist = float(np.clip(by / 2.0 + 0.35, 0.30, 0.55))

    scan_positions = np.array([
        # Top-down: captures base plate top surface and stem top
        [cx,        cy,            joint_top_z + 0.40],
        # Lateral +Y (mid-height): captures the front vertical face of the stem
        [cx,        cy + side_dist, joint_mid_z + 0.05],
        # Lateral -Y (mid-height): captures the rear vertical face of the stem
        [cx,        cy - side_dist, joint_mid_z + 0.05],
        # Diagonal front-right (above + angled): captures base + stem junction
        [cx + 0.25, cy + side_dist * 0.6, joint_mid_z + 0.20],
        # Diagonal back-left (above + angled): captures opposite base + stem junction
        [cx - 0.20, cy - side_dist * 0.6, joint_mid_z + 0.20],
    ])
    # Aim at the actual joint centre (not just table midpoint)
    table_target = np.array([cx, cy, joint_mid_z])
    
    video_frames_dir = os.path.join(DATA_DIR, "video_frames")
    os.makedirs(video_frames_dir, exist_ok=True)
    frame_idx = 0
    
    width, height = 800, 600
    fov, near, far = 60, 0.1, 1.5
    proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)
    
    point_clouds, rgbs, cam_positions, cam_orientations = [], [], [], []
    _log(f"[STEP] SCAN_START total={len(scan_positions)}", log_cb)
    
    trajectory = []
    STATIC_FRAMES = 1
    TRANSITION_FRAMES = 0
    
    for i in range(len(scan_positions)):
        for f in range(STATIC_FRAMES):
            is_scan = (f == STATIC_FRAMES - 1)
            trajectory.append((scan_positions[i], i if is_scan else -1))
            
        if i < len(scan_positions) - 1:
            start_pos = scan_positions[i]
            end_pos = scan_positions[i+1]
            for t in np.linspace(0, 1, TRANSITION_FRAMES, endpoint=False):
                if t == 0: continue
                interp_pos = start_pos * (1 - t) + end_pos * t
                trajectory.append((interp_pos, -1))
                
    for frame_count, (pos, scan_index) in enumerate(trajectory):
        p.stepSimulation()
        
        cam_target = table_target + np.array([0, 0, 0.1])
        look_dir = cam_target - pos
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
        up_vector = [0, 1, 0] if abs(look_dir[2]) > 0.99 else [0, 0, 1]
        
        view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                          cameraTargetPosition=cam_target,
                                          cameraUpVector=up_vector)
                                          
        light_dir = [pos[0] - pos[0], pos[1] - pos[1], 2.0] # Assuming cam_eye is pos
        _, _, rgbImg, depthImg, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, shadow=0, lightDirection=light_dir, lightColor=[1, 1, 1])
        
        rgb_arr = np.reshape(rgbImg, (height, width, 4))[:,:,:3]
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(video_frames_dir, f"{frame_idx:05d}.jpg"), bgr)
        frame_idx += 1
        
        if scan_index != -1:
            pts, clrs = convert_depth_to_pointcloud(depthImg, rgbImg, view_matrix, proj_matrix, width, height, far, near)
            point_clouds.append((pts, clrs))
            rgbs.append(rgb_arr)
            cam_positions.append(pos)
            cam_orientations.append([0,0,0,1])
            _log(f"[STEP] SCAN_POSITION_DONE index={scan_index}", log_cb)
            
    _log("[STEP] SAVING_DATA", log_cb)
    for i, ((pts, clrs), rgb, cam_pos, cam_ori) in enumerate(zip(point_clouds, rgbs, cam_positions, cam_orientations)):
        if len(pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(clrs)
            o3d.io.write_point_cloud(os.path.join(DATA_DIR, f"scan_{i}.pcd"), pcd)
        else:
            o3d.io.write_point_cloud(os.path.join(DATA_DIR, f"scan_{i}.pcd"), o3d.geometry.PointCloud())
            
        np.save(os.path.join(DATA_DIR, f"cam_pos_{i}.npy"), cam_pos)
        np.save(os.path.join(DATA_DIR, f"cam_ori_{i}.npy"), cam_ori)
        img = Image.fromarray(rgb)
        img.save(os.path.join(DATA_DIR, f"rgb_{i}.jpg"))
        
    _log("[STEP] ENCODING_VIDEO", log_cb)
    try:
        # Make sure video height supports standard formats
        subprocess.run([
            "ffmpeg", "-y", "-nostdin", "-framerate", "2", "-i", os.path.join(video_frames_dir, "%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", os.path.join(DATA_DIR, "scan_video.mp4")
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except Exception as e:
        _log(f"    [ERROR] ffmpeg encoding failed: {e}", log_cb)
        
    with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
        
    _log("[STEP] SCAN_COMPLETE", log_cb)

if __name__ == "__main__":
    run_scan()
