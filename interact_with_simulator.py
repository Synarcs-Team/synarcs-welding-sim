from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, FixedCuboid, DynamicCuboid
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.prims import RigidPrim, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.numpy import rot_matrices_to_quats

import numpy as np
import os
# create table
# # Apply material
# assets_root_path = get_assets_root_path()

# if not assets_root_path:
#     print("Error: Could not find assets root path.")
#     return False

# visual_material_url = f"{assets_root_path}/Isaac/Samples/Inventory/Materials/Steel.usd"
# add_reference_to_stage(visual_material_url, f"{parent_path}/{name}/Looks/VisualSteel", prim_type = 'Material')

# binding_api = UsdShade.MaterialBindingAPI.Apply(base.prim)
# binding_api.Bind(UsdShade.Material(prim_utils.get_prim_at_path(f"{parent_path}/{name}/Looks/VisualSteel")))

# binding_api = UsdShade.MaterialBindingAPI.Apply(stem.prim)
# binding_api.Bind(UsdShade.Material(prim_utils.get_prim_at_path(f"{parent_path}/{name}/Looks/VisualSteel")))


def create_table (dimensions: tuple, position: tuple = (0, 0, 0), name: str = "table"):
    table_color = np.array([0.85, 0.85, 1.0])
    table_dimensions = np.array(dimensions)
    table_translation = np.array([table_dimensions[0]/2, table_dimensions[1]/2, table_dimensions[2]/2])

    create_prim(f"/World/{name}", "Xform")
    table_prim = XFormPrim(f"/World/{name}")
    table_prim.set_local_poses(np.array(position).reshape((1, 3)))

    collision_cuboids = []
    RigidPrim(f"/World/{name}")
    table_visual = VisualCuboid(
        prim_path = f"/World/{name}/visual",
        name = 'visual',
        translation = table_translation,
        scale = table_dimensions,
        color = table_color,
        )
    table_collision = FixedCuboid(
        prim_path = f"/World/{name}/collision",
        name = 'collision',
        translation = table_translation,
        scale = table_dimensions,
        visible = False,
        )
    collision_cuboids.append(table_collision)

    return table_prim, collision_cuboids

def create_random_tjoint(width_range: tuple, length_range: tuple, thickness_range: tuple, position: tuple = (0, 0, 0), name: str = "t_joint"):
    

    create_prim(f"/World/{name}", "Xform")
    create_prim(f"/World/{name}/base", "Xform")
    create_prim(f"/World/{name}/stem", "Xform")
    RigidPrim(f"/World/{name}")
    t_joint_prim = XFormPrim(f"/World/{name}")

    random_orientation = np.random.uniform(-np.pi, np.pi)
    orientation = np.array([np.cos(random_orientation/2), 0, 0, np.sin(random_orientation/2)])
    t_joint_prim.set_local_poses(np.array(position).reshape((1, 3)), orientation.reshape((1, 4)))

    # set joint dimensions
    base_w = np.random.uniform(*width_range)
    base_l = np.random.uniform(*length_range)
    base_t = np.random.uniform(*thickness_range)
    stem_h = np.random.uniform(*length_range)
    stem_t = base_t
    stem_w = base_w 
    
    base_dimensions = np.array([base_w, base_l, base_t])
    base_position = np.array([base_dimensions[0]/2, base_dimensions[1]/2, base_dimensions[2]/2])

    stem_dimensions = np.array([stem_w, stem_t, stem_h])
    stem_position = np.array([stem_dimensions[0]/2, stem_dimensions[1]/2 + base_dimensions[1]/2, stem_dimensions[2]/2 + base_dimensions[2]])

    base_color = np.array([0.2, 0.2, 0.2])
    stem_color = np.array([0.2, 0.2, 0.2])

    # create base and stem
    collision_cuboids = []
    base_visual = VisualCuboid(
        prim_path = f"/World/{name}/base/visual",
        name = 'visual',
        translation = base_position,
        scale = base_dimensions,
        color = base_color,
        )
    base_collision = FixedCuboid(
        prim_path = f"/World/{name}/base/collision",
        name = 'collision',
        translation = base_position,
        scale = base_dimensions,
        visible = False,
        )
    stem_visual = VisualCuboid(
        prim_path = f"/World/{name}/stem/visual",
        name = 'visual',
        translation = stem_position,
        scale = stem_dimensions,
        color = stem_color,
        )
    stem_collision = FixedCuboid(
        prim_path = f"/World/{name}/stem/collision",
        name = 'collision',
        translation = stem_position,
        scale = stem_dimensions,
        visible = False,
        )
    collision_cuboids.append(base_collision)
    collision_cuboids.append(stem_collision)
    

    return t_joint_prim, collision_cuboids

def import_robot(usd_path: str, prim_path: str, position: tuple = (0, 0, 0), orientation: tuple = (0, 0, 0, 1)):
    abs_path = os.getcwd() + usd_path
    add_reference_to_stage(abs_path, prim_path)
    robot_prim = XFormPrim(prim_path)
    robot_prim.set_local_poses(np.array(position).reshape((1, 3)), np.array(orientation).reshape((1, 4)))
    return robot_prim

def get_orientation_to_target_y_forward(eoat_pos, target_pos):
    # 1. Calculate direction vector (This will be our Y-axis)
    direction = target_pos - eoat_pos
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0]) # Identity if at same spot
    
    direction /= dist
    
    # 2. Safety Check: Choose a reference "Up" vector
    # Default is World Z. If direction is parallel to Z, use World X.
    ref_up = np.array([0, 0, 1])
    if abs(np.dot(direction, ref_up)) > 0.99:
        ref_up = np.array([1, 0, 0])
    
    # 3. Calculate X-axis (Right)
    # Using Y (direction) and the ref_up
    x_axis = np.cross(direction, ref_up)
    x_axis /= np.linalg.norm(x_axis)
    
    # 4. Calculate Z-axis (True Up)
    # Cross of X and Y gives Z
    z_axis = np.cross(x_axis, direction)
    z_axis /= np.linalg.norm(z_axis)
    
    # 5. Create Rotation Matrix for Y-Forward
    # Column 0: X, Column 1: Y (direction), Column 2: Z
    rot_matrix = np.stack([x_axis, direction, z_axis], axis=1)
    
    # 6. Convert to Quaternion (w, x, y, z)
    quaternion = rot_matrices_to_quats(rot_matrix)
    
    return quaternion

def get_orientation_to_target_x_forward(eoat_pos, target_pos):
    # 1. Calculate direction vector (X-axis)
    direction = target_pos - eoat_pos
    direction /= np.linalg.norm(direction)
    
    # 2. Define a temporary 'Up' to find the 'Right'
    world_up = np.array([0, 0, 1])
    
    # 3. Calculate Y-axis (Right)
    # Cross product of Z (up) and X (direction) gives Y
    y_axis = np.cross(world_up, direction)
    y_axis /= np.linalg.norm(y_axis)
    
    # 4. Calculate Z-axis (True Up)
    # Cross product of X (direction) and Y (right) gives Z
    z_axis = np.cross(direction, y_axis)
    
    # 5. Create Rotation Matrix
    # Column 0: X (direction), Column 1: Y (right), Column 2: Z (up)
    rot_matrix = np.stack([direction, y_axis, z_axis], axis=1)
    
    # 6. Convert to Quaternion (w, x, y, z)
    quaternion = rot_matrices_to_quats(rot_matrix)
    
    return quaternion

world = World().instance()
# reset the world
if world and world.is_playing():
    world.stop()
    print("Simulation reset.")
world.scene.clear()
for _ in range(10):
    world.step(render=True)
world.scene.add_default_ground_plane()


table_args = {
    "dimensions": (1.5, 3, 1),
    "position": (0, -1.5, 0),
    "name": "table"
}
t_joint_args = {
    "width_range": (0.1, 0.2),
    "length_range": (0.1, 0.2),
    "thickness_range": (0.02, 0.03),
    "name": "t_joint",
    "position": (0.75, 0, table_args["dimensions"][2])
}

# create scene
seed = 0
np.random.seed(seed)
table_prim, table_collision_cuboids = create_table(**table_args)
t_joint_prim, t_joint_collision_cuboids = create_random_tjoint(**t_joint_args)
collision_space = VisualCuboid(
    prim_path = "/World/collision",
    name = "collision_cuboid",
    position = (0.9, 0, 1.15),
    scale = (0.5, 0.5, 0.3),
    visible = False,
    color = np.array([1, 0, 0])
)
# setup robot
robot_position = (0.2, 0, table_args["dimensions"][2])
robot_orientation = (0, 0, 0, 1)
robot_prim = import_robot("/world/ur10_w_realsense.usd", "/World/ur10", robot_position, robot_orientation)
manipulator = SingleManipulator(
    prim_path="/World/ur10",
    name="ur10_robot",
    end_effector_prim_path="/World/ur10/ur10_w_realsense/ur10/ee_link",
)


from isaacsim.sensors.camera import Camera
camera = Camera(
    prim_path = "/World/ur10/ur10_w_realsense/ur10/ee_link/rsd455/RSD455/Camera_Pseudo_Depth"
)
camera_prim = XFormPrim("/World/ur10/ur10_w_realsense/ur10/ee_link/rsd455/RSD455/Camera_Pseudo_Depth")
# print(camera_prim.get_world_poses()[0].flatten())
# print(camera_prim.get_world_poses()[1].flatten())
# reset world to register scene
world.reset()
world.play()
for _ in range(10): 
    world.step(render=True)
world.pause()
manipulator.initialize()
camera.initialize()
camera.set_resolution((640, 480))
camera.add_pointcloud_to_frame()

from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver, LulaCSpaceTrajectoryGenerator, ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.core.utils.types import ArticulationAction


# --- 1. SETUP CONFIGURATION PATHS ---
# We use the built-in UR10 configs provided by the extension
from isaacsim.robot_motion.motion_generation import interface_config_loader
config = interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
urdf_path = config["urdf_path"]
robot_description_path = config["robot_description_path"]
ee_name = config["end_effector_frame_name"]

# setup kinematic solver
kin_solver = LulaKinematicsSolver(
    robot_description_path=robot_description_path,
    urdf_path=urdf_path
)
ik_solver = ArticulationKinematicsSolver(
    robot_articulation=manipulator,
    kinematics_solver=kin_solver,
    end_effector_frame_name=ee_name 
)

# print(ik_solver.compute_end_effector_pose())

#setup rrt path planner
planner = RRT(
    robot_description_path=robot_description_path,
    urdf_path=urdf_path,
    rrt_config_path="./config/rrt_config.yaml",
    end_effector_frame_name=ee_name
)
planner.set_robot_base_pose(np.array(robot_position), np.array(robot_orientation))
planner.add_cuboid(collision_space, static = True)
planner.add_cuboid(table_collision_cuboids[0], static = True)
planner.update_world()
# setup trajectory generator
cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
    robot_description_path=robot_description_path,
    urdf_path=urdf_path,
)

# for obstacle in obstacles:
#     print(obstacle.prim.IsValid())

test_number = 6
# plan and execute paths
desired_positions = np.array([
    [0.5, 0.50, 1.5],
    [1.1, 0.5, 1.5],
    [1.1, 0, 1.5],
    [1.1, -0.50, 1.5],
    [0.5, -0.50, 1.5],
])
table_target = np.array([0.9, 0, 1.15])
camera.set_clipping_range(far_distance = 1)
point_clouds = []
rgbs = []
camera_positions = []
camera_orientations = []
for position_number in range(len(desired_positions)):
    desired_ori_quat = get_orientation_to_target_x_forward(desired_positions[position_number], table_target)
    planner.set_end_effector_target(desired_positions[position_number], desired_ori_quat)
    active_joint_positions = manipulator.get_joint_positions()
    watched_joint_positions = np.array([])
    path = planner.compute_path(active_joint_positions, watched_joint_positions)

    if path is None:
        print(f"No path found for position:{desired_positions[position_number]}")
    else:
        trajectory = cspace_trajectory_generator.compute_c_space_trajectory(path)

        articulation_trajectory = ArticulationTrajectory(
        robot_articulation=manipulator,
        trajectory=trajectory,
        physics_dt=1/60,
    )
        action_sequence = articulation_trajectory.get_action_sequence()

        world.play()
        for action in action_sequence:
            manipulator.apply_action(action)
            world.step(render=True)
        world.pause()

        frame = camera.get_current_frame()
        point_clouds.append(frame["pointcloud"]["data"])
        rgbs.append(frame["rgb"][:,:,:3])
        camera_positions.append(camera_prim.get_world_poses()[0].flatten())
        camera_orientations.append(camera_prim.get_world_poses()[1].flatten())

import open3d as o3d
# from PIL import Image
# base_name = "./data"
# import os
# os.mkdir(f"{base_name}/test_{test_number}")
# for position_number in range(len(desired_positions)):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_clouds[position_number])
#     o3d.io.write_point_cloud(f"{base_name}/test_{test_number}/test_{test_number}_pcd_{position_number}.pcd", pcd)

#     np.save(f"{base_name}/test_{test_number}/test_{test_number}_camera_position_{position_number}.npy", camera_positions[position_number])
#     np.save(f"{base_name}/test_{test_number}/test_{test_number}_camera_orientation_{position_number}.npy", camera_orientations[position_number])

#     img = Image.fromarray(rgbs[position_number])
#     img.save(f"{base_name}/test_{test_number}/test_{test_number}_rgb_{position_number}.jpg")



# weld seam detection
merged_pcd = o3d.geometry.PointCloud()
for pcd in point_clouds:
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(pcd)
    merged_pcd += pcd_temp

min_bound = np.array([0.65, -0.25, 1.01])
max_bound = np.array([1.15, 0.25, 1.3])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
bbox.color = (1, 0, 0)

cropped_pcd = merged_pcd.crop(bbox)
world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

cropped_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=20))
cropped_pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 3])
cropped_pcd.paint_uniform_color([0, 0, 1])

# o3d.visualization.draw_geometries([cropped_pcd, world_origin])
from t_joint_planning import * 
weld_segment_1, weld_segment_2, weld_segment_1_points, weld_segment_2_points = find_t_joint_paths(cropped_pcd, table_normal=np.array([0, 0, 1]), angle_threshold_deg=10, visualize=False)
print(weld_segment_1_points)
# o3d.visualization.draw_geometries([cropped_pcd, weld_segment_1, weld_segment_2, world_origin])


# show weld seams
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
colors = [(1.0, 0.0, 0.0, 1.0)] 
sizes = [10.0] 

start_point = [tuple(weld_segment_1_points[0])]
end_point = [tuple(weld_segment_1_points[1])]
draw.draw_lines(start_point, end_point, colors, sizes)

start_point = [tuple(weld_segment_2_points[0])]
end_point = [tuple(weld_segment_2_points[1])]
draw.draw_lines(start_point, end_point, colors, sizes)

# execute weld seams
def get_straight_line_path(start_pose, end_pose, steps=20):
    """
    start_pose: (pos, ori)
    end_pose: (pos, ori)
    """
    start_pos, start_ori = start_pose
    end_pos, end_ori = end_pose
    
    joint_path = []
    
    for i in range(steps):
        # 1. Linear interpolation of position (LERP)
        alpha = i / (steps - 1)
        interp_pos = start_pos + alpha * (end_pos - start_pos)
        
        # 2. Spherically interpolate orientation (SLERP)
        # Note: You can use scipy.spatial.transform.Slerp or manual interpolation
        # For simple cases, keep start_ori if orientation doesn't change
        
        # 3. Solve IK for this specific waypoint
        # ik_solver should be the ArticulationKinematicsSolver you already set up
        joint_targets, success = ik_solver.compute_inverse_kinematics(
            target_position=interp_pos,
            target_orientation=end_ori
        )
        
        if success:
            joint_path.append(joint_targets)
        else:
            print(f"Warning: IK failed at step {i}")
            
    return np.array(joint_path)

print(t_joint_collision_cuboids[1].prim.IsValid())
planner.disable_obstacle(collision_space)
planner.add_cuboid(t_joint_collision_cuboids[0], static = True)
planner.add_cuboid(t_joint_collision_cuboids[1], static = True)
planner.update_world()
desired_positions = np.array([
    [0.5, -0.50, 1.5],
    weld_segment_1_points[0].flatten(),
    weld_segment_1_points[1].flatten(),
    weld_segment_2_points[0].flatten(),
    weld_segment_2_points[1].flatten(),
    [0.5, -0.50, 1.5],
])
desired_positions = np.array([
    [0.5, -0.50, 1.5],
])
steps = 10
for i in range(steps):
    alpha = i / (steps - 1)
    interp_pos = weld_segment_1_points[0].flatten() + alpha * (weld_segment_1_points[1].flatten() - weld_segment_1_points[0].flatten())
    desired_positions = np.vstack((desired_positions, interp_pos.reshape(1, -1)))
for i in range(steps):
    alpha = i / (steps - 1)
    interp_pos = weld_segment_2_points[0].flatten() + alpha * (weld_segment_2_points[1].flatten() - weld_segment_2_points[0].flatten())
    desired_positions = np.vstack((desired_positions, interp_pos.reshape(1, -1)))
desired_positions = np.vstack((desired_positions, np.array([[0.5, -0.50, 1.5]])))
for position_number in range(len(desired_positions)):
    desired_ori_quat = get_orientation_to_target_x_forward(desired_positions[position_number], table_target)
    planner.set_end_effector_target(desired_positions[position_number], desired_ori_quat)
    active_joint_positions = manipulator.get_joint_positions()
    watched_joint_positions = np.array([])
    path = planner.compute_path(active_joint_positions, watched_joint_positions)

    if path is None:
        print(f"No path found for position:{desired_positions[position_number]}")
    else:
        trajectory = cspace_trajectory_generator.compute_c_space_trajectory(path)

        articulation_trajectory = ArticulationTrajectory(
        robot_articulation=manipulator,
        trajectory=trajectory,
        physics_dt=1/60,
    )
        action_sequence = articulation_trajectory.get_action_sequence()

        world.play()
        for action in action_sequence:
            manipulator.apply_action(action)
            world.step(render=True)
        world.pause()
