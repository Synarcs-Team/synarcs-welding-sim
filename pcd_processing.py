import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

from t_joint_planning import *

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def save_pcd(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def merge_point_clouds(pcd_list, positions, quaternions):
    """
    pcd_list: List of o3d.geometry.PointCloud objects
    positions: List of [x, y, z] arrays
    quaternions: List of [x, y, z, w] arrays
    """
    merged_pcd = o3d.geometry.PointCloud()
    
    for pcd, pos, quat in zip(pcd_list, positions, quaternions):
        # 1. Create a 4x4 Identity Matrix
        trans_mat = np.eye(4)
        
        # 2. Convert Quaternion to 3x3 Rotation Matrix
        # Scipy uses [x, y, z, w] format
        rotation_matrix = R.from_quat(quat).as_matrix()
        
        # 3. Fill the 4x4 Matrix
        trans_mat[:3, :3] = rotation_matrix
        trans_mat[:3, 3] = pos
        
        # 4. Transform a copy of the cloud to global space
        # We use a copy so the original object remains unchanged
        pcd_transformed = copy.deepcopy(pcd)
        pcd_transformed.transform(trans_mat)
        
        # 5. Add to the merged object
        merged_pcd += pcd_transformed
        
    return merged_pcd

def visualize_poses(positions, quaternions, axis_size=10.0):
    """
    positions: List or array of [x, y, z]
    quaternions: List or array of [x, y, z, w]
    axis_size: Length of the coordinate frame axes
    """
    visual_elements = []

    for pos, quat in zip(positions, quaternions):
        
        # 1. Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        
        # 2. Convert quaternion to rotation matrix
        rot_mat = R.from_quat(quat).as_matrix()
        
        # 3. Apply orientation and position
        frame.rotate(rot_mat, center=(0, 0, 0))
        frame.translate(pos)
        
        visual_elements.append(frame)
    
    # Add a global world origin for reference (optional)
    world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size*2)
    visual_elements.append(world_origin)

    o3d.visualization.draw_geometries(visual_elements)

def main():

    test_number = 6
    n_scans = 5
    folder_name = f"./data/test_{test_number}"

    pcds = []
    positions = []
    orientations = []

    for scan_number in range(n_scans):
        file_name = folder_name + f"/test_{test_number}_pcd_{scan_number}.pcd"
        pcds.append(load_pcd(file_name))

        file_name = folder_name + f"/test_{test_number}_camera_position_{scan_number}.npy"
        positions.append(np.load(file_name))

        file_name = folder_name + f"/test_{test_number}_camera_orientation_{scan_number}.npy"
        orientations.append(np.load(file_name))

    world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged_pcd += pcd

    min_bound = np.array([0.65, -0.25, 1.01])
    max_bound = np.array([1.15, 0.25, 1.3])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = (1, 0, 0)

    cropped_pcd = merged_pcd.crop(bbox)
    
    # o3d.visualization.draw_geometries([cropped_pcd, world_origin])

    # estimate normals
    cropped_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=20))
    cropped_pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 3])
    cropped_pcd.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([cropped_pcd, world_origin])

    weld_segment_1, weld_segment_2 = find_t_joint_paths(cropped_pcd, table_normal=np.array([0, 0, 1]), angle_threshold_deg=10, visualize=True)

    o3d.visualization.draw_geometries([cropped_pcd, weld_segment_1, weld_segment_2, world_origin])

    print("done")

if __name__ == "__main__":
    main()