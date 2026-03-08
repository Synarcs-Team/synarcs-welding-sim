"""
sim_weld.py — Stage 4: Welding Execution
Starts Isaac Sim, rebuilds scene, then moves the robot along the
detected weld seam paths loaded from data/latest/seams.json.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

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
from isaacsim.util.debug_draw import _debug_draw

import json, numpy as np

from pathlib import Path
ROOT     = str(Path(__file__).resolve().parents[5])
DATA_DIR = os.path.join(ROOT, "data", "latest")

# ── Load seams and config ─────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "seams.json")) as f:
    seams = json.load(f)
with open(os.path.join(DATA_DIR, "config.json")) as f:
    cfg = json.load(f)

seg1_start = np.array(seams["seam1"]["start"])
seg1_end   = np.array(seams["seam1"]["end"])
seg2_start = np.array(seams["seam2"]["start"])
seg2_end   = np.array(seams["seam2"]["end"])
print("[STEP] SEAMS_LOADED", flush=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def create_table(dims, position, name):
    color = np.array([0.85,0.85,1.0])
    d = np.array(dims); t = d/2
    create_prim(f"/World/{name}","Xform"); prim=XFormPrim(f"/World/{name}")
    prim.set_local_poses(np.array(position).reshape((1,3))); RigidPrim(f"/World/{name}")
    VisualCuboid(prim_path=f"/World/{name}/visual",name="visual",translation=t,scale=d,color=color)
    c=FixedCuboid(prim_path=f"/World/{name}/collision",name="collision",translation=t,scale=d,visible=False)
    return prim,[c]

def create_tjoint(base_width,base_length,base_thickness,stem_height,stem_thickness,position,seed,name):
    create_prim(f"/World/{name}","Xform"); create_prim(f"/World/{name}/base","Xform"); create_prim(f"/World/{name}/stem","Xform")
    RigidPrim(f"/World/{name}"); prim=XFormPrim(f"/World/{name}")
    rng=np.random.default_rng(seed); a=rng.uniform(-np.pi,np.pi)
    ori=np.array([np.cos(a/2),0,0,np.sin(a/2)])
    prim.set_local_poses(np.array(position).reshape((1,3)),ori.reshape((1,4)))
    bd=np.array([base_width,base_length,base_thickness]); bp=bd/2
    sd=np.array([base_width,stem_thickness,stem_height])
    sp=np.array([base_width/2, stem_thickness/2+base_length/2, stem_height/2+base_thickness])
    color=np.array([0.2,0.2,0.2])
    VisualCuboid(prim_path=f"/World/{name}/base/visual",name="visual",translation=bp,scale=bd,color=color)
    bc=FixedCuboid(prim_path=f"/World/{name}/base/collision",name="collision",translation=bp,scale=bd,visible=False)
    VisualCuboid(prim_path=f"/World/{name}/stem/visual",name="visual",translation=sp,scale=sd,color=color)
    sc=FixedCuboid(prim_path=f"/World/{name}/stem/collision",name="collision",translation=sp,scale=sd,visible=False)
    return prim,[bc,sc]

def get_orientation_to_target_x_forward(eoat_pos, target_pos):
    d=target_pos-eoat_pos; d/=np.linalg.norm(d)
    y=np.cross(np.array([0,0,1]),d); y/=np.linalg.norm(y)
    z=np.cross(d,y)
    return rot_matrices_to_quats(np.stack([d,y,z],axis=1))

# ── Build world ───────────────────────────────────────────────────────────────
world = World().instance()
if world and world.is_playing(): world.stop()
world.scene.clear()
for _ in range(10): world.step(render=True)
world.scene.add_default_ground_plane()

table_dims = (1.5, 3, 1)
table_prim, table_colls = create_table(table_dims, (0,-1.5,0), "table")
t_joint_prim, t_joint_colls = create_tjoint(
    cfg["base_width"],cfg["base_length"],cfg["base_thickness"],
    cfg["stem_height"],cfg["stem_thickness"],
    (0.75,0,table_dims[2]),cfg["seed"],"t_joint"
)
collision_space = VisualCuboid(
    prim_path="/World/collision",name="collision_cuboid",
    position=(0.9,0,1.15),scale=(0.5,0.5,0.3),visible=False,color=np.array([1,0,0])
)
robot_pos=(0.2,0,table_dims[2]); robot_ori=(0,0,0,1)
add_reference_to_stage(os.path.join(ROOT,"world","ur10_w_realsense.usd"),"/World/ur10")
rp=XFormPrim("/World/ur10")
rp.set_local_poses(np.array(robot_pos).reshape((1,3)),np.array(robot_ori).reshape((1,4)))
manipulator=SingleManipulator(prim_path="/World/ur10",name="ur10_robot",
    end_effector_prim_path="/World/ur10/ur10_w_realsense/ur10/ee_link")

world.reset(); world.play()
for _ in range(10): world.step(render=True)
world.pause(); manipulator.initialize()

# Add a camera for recording the weld process from a fixed perspective
import isaacsim.core.utils.prims as prim_utils
cam_path = "/World/Camera"
prim_utils.create_prim(cam_path, "Camera", translation=(1.5, -1.0, 1.8), orientation=rot_matrices_to_quats(np.stack([[-1,0,0],[0,-1,0],[0,0,1]], axis=1))) # Approximate look at table

rec_camera = Camera(prim_path=cam_path, resolution=(800, 600))
rec_camera.initialize()

# Prepare for video recording
video_frames_dir = os.path.join(DATA_DIR, "video_frames")
os.makedirs(video_frames_dir, exist_ok=True)
frame_idx = 0

mc=interface_config_loader.load_supported_motion_policy_config("UR10","RMPflow")
kin_solver=LulaKinematicsSolver(mc["robot_description_path"],mc["urdf_path"])
ik_solver=ArticulationKinematicsSolver(manipulator,kin_solver,mc["end_effector_frame_name"])
planner=RRT(mc["robot_description_path"],mc["urdf_path"],
            rrt_config_path="./config/rrt_config.yaml",
            end_effector_frame_name=mc["end_effector_frame_name"])
planner.set_robot_base_pose(np.array(robot_pos),np.array(robot_ori))
planner.disable_obstacle(collision_space)
planner.add_cuboid(table_colls[0],static=True)
planner.add_cuboid(t_joint_colls[0],static=True)
planner.add_cuboid(t_joint_colls[1],static=True)
planner.update_world()
traj_gen=LulaCSpaceTrajectoryGenerator(mc["robot_description_path"],mc["urdf_path"])

# ── Draw weld seams in viewport ───────────────────────────────────────────────
draw=_debug_draw.acquire_debug_draw_interface()
colors=[(1.0,0.0,0.0,1.0)]; sizes=[10.0]
draw.draw_lines([tuple(seg1_start)],[tuple(seg1_end)],colors,sizes)
draw.draw_lines([tuple(seg2_start)],[tuple(seg2_end)],colors,sizes)

# ── Build waypoint list: home → seam1 → seam2 → home ─────────────────────────
STEPS=10
home=np.array([0.5,-0.50,1.5])
waypoints=[home]
for i in range(STEPS):
    a=i/(STEPS-1)
    waypoints.append(seg1_start+a*(seg1_end-seg1_start))
for i in range(STEPS):
    a=i/(STEPS-1)
    waypoints.append(seg2_start+a*(seg2_end-seg2_start))
waypoints.append(home)
waypoints=np.array(waypoints)

table_target=np.array([0.9,0,1.15])
print(f"[STEP] WELD_START total={len(waypoints)}", flush=True)

for i, pos in enumerate(waypoints):
    ori=get_orientation_to_target_x_forward(pos,table_target)
    planner.set_end_effector_target(pos,ori)
    path=planner.compute_path(manipulator.get_joint_positions(),np.array([]))
    if path is None:
        print(f"[WARN] No path for waypoint {i}", flush=True)
        continue
    traj=traj_gen.compute_c_space_trajectory(path)
    art_traj=ArticulationTrajectory(manipulator,traj,physics_dt=1/60)
    world.play()
    for action in art_traj.get_action_sequence():
        manipulator.apply_action(action); world.step(render=True)
        
        # Capture frame for video
        rgba = rec_camera.get_rgba()
        if rgba is not None and rgba.shape[0] > 0:
            rgb = rgba[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(video_frames_dir, f"{frame_idx:05d}.jpg"), bgr)
            frame_idx += 1

    world.pause()
    print(f"[STEP] WELD_WAYPOINT_DONE index={i}", flush=True)

# Encode video from captured frames using ffmpeg
import subprocess
print("[STEP] ENCODING_VIDEO", flush=True)
try:
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-framerate", "30", "-i", os.path.join(video_frames_dir, "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(DATA_DIR, "weld_video.mp4")
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
except Exception as e:
    print(f"    [ERROR] ffmpeg encoding failed: {e}", flush=True)

print("[STEP] WELD_COMPLETE", flush=True)
simulation_app.close()
