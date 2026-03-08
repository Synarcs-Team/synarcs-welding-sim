"""
sim_scan.py — Stage 1: Scanning
Starts Isaac Sim, builds scene, moves robot to 5 camera positions,
captures point clouds + RGB images, saves them to data/latest/.
Prints progress markers for the web app to parse.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)   # flush every line for streaming

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "enable_livestream": True})

from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
import cv2
from isaacsim.core.api.objects import VisualCuboid, FixedCuboid
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.prims import RigidPrim, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.numpy import rot_matrices_to_quats
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver, ArticulationKinematicsSolver,
    LulaCSpaceTrajectoryGenerator, ArticulationTrajectory,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT

import json, numpy as np
from PIL import Image
import open3d as o3d

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
from welding_simulator.core.joint_factory import create_joint_from_config

def create_table(dimensions, position=(0,0,0), name="table"):
    table_color    = np.array([0.85, 0.85, 1.0])
    table_dims     = np.array(dimensions)
    table_trans    = table_dims / 2
    create_prim(f"/World/{name}", "Xform")
    prim = XFormPrim(f"/World/{name}")
    prim.set_local_poses(np.array(position).reshape((1,3)))
    RigidPrim(f"/World/{name}")
    VisualCuboid(prim_path=f"/World/{name}/visual", name="visual",
                 translation=table_trans, scale=table_dims, color=table_color)
    coll = FixedCuboid(prim_path=f"/World/{name}/collision", name="collision",
                       translation=table_trans, scale=table_dims, visible=False)
    return prim, [coll]

def get_orientation_to_target_x_forward(eoat_pos, target_pos):
    direction  = target_pos - eoat_pos
    direction /= np.linalg.norm(direction)
    world_up   = np.array([0, 0, 1])
    y_axis     = np.cross(world_up, direction)
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis /= norm_y
    z_axis     = np.cross(direction, y_axis)
    rot_matrix = np.stack([direction, y_axis, z_axis], axis=1)
    return rot_matrices_to_quats(rot_matrix)

# ── Build world ───────────────────────────────────────────────────────────────
world = World().instance()
if world and world.is_playing():
    world.stop()
world.scene.clear()
for _ in range(10): world.step(render=True)
world.scene.add_default_ground_plane()

table_dims = (1.5, 3, 1)
table_pos  = (0, -1.5, 0)
table_prim, table_colls = create_table(table_dims, table_pos)

# Build Dynamic Joint
joint_prim, joint_colls, joint_bbox = create_joint_from_config(
    cfg, position=(0.75, 0, table_dims[2])
)
collision_space = VisualCuboid(
    prim_path="/World/collision", name="collision_cuboid",
    position=(0.9, 0, 1.15), scale=(0.5, 0.5, 0.3),
    visible=False, color=np.array([1,0,0])
)
robot_pos = (0.2, 0, table_dims[2])
robot_ori = (0, 0, 0, 1)
add_reference_to_stage(os.path.join(ROOT, "world", "ur10_w_realsense.usd"), "/World/ur10")
robot_prim = XFormPrim("/World/ur10")
robot_prim.set_local_poses(np.array(robot_pos).reshape((1,3)), np.array(robot_ori).reshape((1,4)))

manipulator = SingleManipulator(
    prim_path="/World/ur10", name="ur10_robot",
    end_effector_prim_path="/World/ur10/ur10_w_realsense/ur10/ee_link",
)
camera      = Camera(prim_path="/World/ur10/ur10_w_realsense/ur10/ee_link/rsd455/RSD455/Camera_Pseudo_Depth")
camera_prim = XFormPrim("/World/ur10/ur10_w_realsense/ur10/ee_link/rsd455/RSD455/Camera_Pseudo_Depth")

world.reset(); world.play()
for _ in range(10): world.step(render=True)
world.pause()
manipulator.initialize()
camera.initialize()
camera.set_resolution((800, 600))
camera.add_pointcloud_to_frame()

# Prepare for video recording
video_frames_dir = os.path.join(DATA_DIR, "video_frames")
os.makedirs(video_frames_dir, exist_ok=True)
frame_idx = 0

# ── Motion planning setup ─────────────────────────────────────────────────────
mc = interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
kin_solver = LulaKinematicsSolver(mc["robot_description_path"], mc["urdf_path"])
ik_solver  = ArticulationKinematicsSolver(manipulator, kin_solver, mc["end_effector_frame_name"])
planner    = RRT(mc["robot_description_path"], mc["urdf_path"],
                 rrt_config_path="./config/rrt_config.yaml",
                 end_effector_frame_name=mc["end_effector_frame_name"])
planner.set_robot_base_pose(np.array(robot_pos), np.array(robot_ori))
planner.add_cuboid(collision_space, static=True)
planner.add_cuboid(table_colls[0], static=True)
planner.update_world()
traj_gen = LulaCSpaceTrajectoryGenerator(mc["robot_description_path"], mc["urdf_path"])

# ── Scan loop ─────────────────────────────────────────────────────────────────
# Dynamically compute camera positions in the UR10's reachable workspace.
# Original working positions were near: x∈[0.5,1.1], y∈[-0.5,0.5], z=1.5
# The joint is placed at (0.75, 0, table_top). We keep all positions in the
# +X half so the arm doesn't have to reach behind its own base.
bx, by, bz = joint_bbox
cx, cy = 0.75, 0.0                      # Joint center on the table
z_scan = table_dims[2] + bz + 0.35     # Height: table surface + joint + clearance

# Y spread scaled by the joint width, clamped to safe arm workspace
y_offset = float(np.clip(by / 2.0 + 0.25, 0.25, 0.50))

scan_positions = np.array([
    [cx,      cy,          z_scan + 0.15],  # Top-down (directly above)
    [cx + 0.35, cy + y_offset, z_scan],     # Front-right
    [cx + 0.35, cy - y_offset, z_scan],     # Front-left
    [cx - 0.25, cy + y_offset, z_scan],     # Slight back-right (still in reach)
    [cx - 0.25, cy - y_offset, z_scan],     # Slight back-left
])
table_target = np.array([cx, cy, table_dims[2] + bz / 2.0])
camera.set_clipping_range(far_distance=1.5)

point_clouds, rgbs, cam_positions, cam_orientations = [], [], [], []
print(f"[STEP] SCAN_START total={len(scan_positions)}", flush=True)

for i, pos in enumerate(scan_positions):
    ori      = get_orientation_to_target_x_forward(pos, table_target)
    planner.set_end_effector_target(pos, ori)
    path     = planner.compute_path(manipulator.get_joint_positions(), np.array([]))
    if path is None:
        print(f"[WARN] No path for position {i}", flush=True)
        continue
    traj     = traj_gen.compute_c_space_trajectory(path)
    art_traj = ArticulationTrajectory(manipulator, traj, physics_dt=1/60)
    world.play()
    for action in art_traj.get_action_sequence():
        manipulator.apply_action(action)
        world.step(render=True)
        
        # Capture frame for video
        rgba = camera.get_rgba()
        if rgba is not None and rgba.shape[0] > 0:
            rgb = rgba[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(video_frames_dir, f"{frame_idx:05d}.jpg"), bgr)
            frame_idx += 1

    world.pause()
    frame = camera.get_current_frame()
    point_clouds.append(frame["pointcloud"]["data"])
    rgbs.append(frame["rgb"][:,:,:3])
    cam_positions.append(camera_prim.get_world_poses()[0].flatten())
    cam_orientations.append(camera_prim.get_world_poses()[1].flatten())
    print(f"[STEP] SCAN_POSITION_DONE index={i}", flush=True)

# ── Save outputs ──────────────────────────────────────────────────────────────
print("[STEP] SAVING_DATA", flush=True)
for i, (pcd_data, rgb, cam_pos, cam_ori) in enumerate(zip(point_clouds, rgbs, cam_positions, cam_orientations)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data)
    o3d.io.write_point_cloud(os.path.join(DATA_DIR, f"scan_{i}.pcd"), pcd)
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
simulation_app.close()
os._exit(0)
