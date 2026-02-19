import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
# from src.preprocessing.background_removal import create_mesh_from_plane
import copy

def create_mesh_from_plane(plane_model, size=1.0):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    
    # 1. Create a coordinate frame to find the rotation
    # We want to align the Z-axis [0,0,1] with our plane normal
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis = cross product of Z and Normal
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c_val = np.dot(z_axis, normal)
    
    # Skew-symmetric matrix
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # Rodrigues' rotation formula
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))

    # 2. Create the mesh (a thin box)
    mesh_plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)
    
    # 3. Center the mesh
    mesh_plane.translate(-np.array([size/2, size/2, 0.0005]))
    
    # 4. Rotate to align with normal
    mesh_plane.rotate(R, center=(0, 0, 0))
    
    # 5. Translate to the correct distance 'd'
    # The point on the plane closest to origin is -d * normal
    translation = -d * normal
    mesh_plane.translate(translation)
    
    # Give it a color (e.g., semi-transparent blue or solid grey)
    mesh_plane.paint_uniform_color([0.2, 0.5, 0.9]) # Blueish
    
    return mesh_plane

def get_cuboid_error(params, points):
    # params: [x, y, z, roll, pitch, yaw, length, width, height]
    pos, angles, dims = params[:3], params[3:6], params[6:]
    
    # 1. Transform points to local box coordinate system
    rot_mat = R.from_euler('xyz', angles).as_matrix()
    local_pts = (points - pos) @ rot_mat
    
    # 2. Calculate distance to box surface
    # Find distance from the center along each axis relative to half-extents
    d = np.abs(local_pts) - (dims / 2)
    
    # Points outside the box (positive d)
    external_dist = np.linalg.norm(np.maximum(d, 0), axis=1)**2
    
    # Points inside the box (negative d) - distance to closest face
    internal_dist = np.where(np.all(d < 0, axis=1), np.max(d, axis=1)**2, 0)
    
    return np.sum(external_dist + internal_dist)

def huber_loss(distances, delta=1.0):
    """
    delta: The distance threshold. Errors larger than delta are treated linearly.
    """
    abs_dist = np.abs(distances)
    quadratic = np.minimum(abs_dist, delta)
    linear = abs_dist - quadratic
    return 0.5 * quadratic**2 + delta * linear

def get_robust_cuboid_error(params, points, delta=0.01):
    pos, angles, dims = params[:3], params[3:6], params[6:]
    rot_mat = R.from_euler('xyz', angles).as_matrix()
    local_pts = (points - pos) @ rot_mat
    
    # Calculate signed distances to box boundaries
    d = np.abs(local_pts) - (dims / 2)
    
    # External distance (points outside)
    ext_d = np.linalg.norm(np.maximum(d, 0), axis=1)
    # Internal distance (points inside)
    int_d = np.where(np.all(d < 0, axis=1), np.max(d, axis=1), 0)
    
    total_distances = ext_d + np.abs(int_d)
    
    # Apply Huber Loss instead of np.sum(total_distances**2)
    return np.sum(huber_loss(total_distances, delta=delta))
    
def fit_cuboid(pcd, initial_guess=None):
    points = np.asarray(pcd.points)
    
    if initial_guess is None:
        center = pcd.get_center()
        extent = pcd.get_max_bound() - pcd.get_min_bound()
        # Initial guess: [x, y, z, r, p, y, L, W, H]
        initial_guess = [*center, 0, 0, 0, *extent]

    # Optimization with bounds to prevent negative dimensions
    bounds = [(None, None)]*6 + [(0.001, None)]*3 
    res = minimize(get_robust_cuboid_error, initial_guess, args=(points,), 
                   method='L-BFGS-B', bounds=bounds)
    
    return res.x

def create_fitted_box_mesh(params, color=[0, 1, 0], opacity=0.5):
    """
    Creates a mesh based on the optimized cuboid parameters.
    params: [x, y, z, roll, pitch, yaw, length, width, height]
    """
    pos = params[:3]
    angles = params[3:6]
    l, w, h = params[6:]

    # 1. Create a centered box mesh
    # Open3D creates boxes starting at (0,0,0) and extending to (+L, +W, +H)
    # We create it with size 1 and scale it to ensure it centers correctly
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
    
    # 2. Shift the mesh so its center is at (0,0,0) before rotating
    mesh_box.translate(-mesh_box.get_center())

    # 3. Rotate the mesh
    rot_mat = R.from_euler('xyz', angles).as_matrix()
    mesh_box.rotate(rot_mat, center=(0, 0, 0))

    # 4. Translate to the fitted position
    mesh_box.translate(pos)

    # 5. Styling
    mesh_box.paint_uniform_color(color)
    # Note: Open3D visualization of opacity requires specialized shaders, 
    # but we can return the mesh for standard viewing.
    
    return mesh_box

def find_similar_points(pcd, target_normal, angle_threshold_deg):
    # 1. Extract normals
    normals = np.asarray(pcd.normals)

    # 2. Normalize target vector (just in case)
    target_normal = target_normal / np.linalg.norm(target_normal)

    # 3. Calculate dot products
    # Using absolute if you want to include opposite directions (optional)
    similarities = np.dot(normals, target_normal)

    # 4. Filter by cosine of the threshold angle
    min_similarity = np.cos(np.radians(angle_threshold_deg))
    indices = np.where(similarities > min_similarity)[0]

    pcd_similar = pcd.select_by_index(indices)
    pcd_similar.paint_uniform_color([1, 0, 0])
    pcd_different = pcd.select_by_index(indices, invert=True)
    pcd_different.paint_uniform_color([0, 1, 0])

    print("Red are similar, green arent")

    return indices, pcd_similar, pcd_different

# line of intersection of two planes
def intersect_planes(p1, p2):
    """
    p1, p2: Plane models as [a, b, c, d]
    """
    n1, d1 = np.array(p1[:3]), p1[3]
    n2, d2 = np.array(p2[:3]), p2[3]

    # 1. Find direction vector
    v = np.cross(n1, n2)
    
    if np.linalg.norm(v) < 1e-6:
        return None, None # Planes are parallel

    # 2. Find a point on the line
    # We solve the system assuming z=0
    # If the line is parallel to the XY plane, we would need to set x=0 or y=0 instead
    A = np.array([n1[:2], n2[:2]])
    B = np.array([-d1, -d2])
    
    try:
        # Solve for x and y
        p_xy = np.linalg.solve(A, B)
        p0 = np.array([p_xy[0], p_xy[1], 0.0])
    except np.linalg.LinAlgError:
        # If z=0 doesn't work, the line is parallel to the XY plane; try x=0
        A = np.array([n1[1:], n2[1:]])
        p_yz = np.linalg.solve(A, B)
        p0 = np.array([0.0, p_yz[0], p_yz[1]])

    return p0, v

def get_centered_intersection_line(p1_model, p2_model, cluster1_pcd, cluster2_pcd):
    # 1. Get the raw mathematical line (p0 might be far away)
    p0_raw, v = intersect_planes(p1_model, p2_model)
    v = v / np.linalg.norm(v) # Ensure direction is a unit vector

    # 2. Find a realistic anchor point
    # We take the average of the two cluster centroids
    center_c1 = cluster1_pcd.get_center()
    center_c2 = cluster2_pcd.get_center()
    q = (center_c1 + center_c2) / 2

    # 3. Project that center 'q' onto the infinite line
    # This gives us a p0 that is physically located at the joint
    p0_anchored = p0_raw + np.dot(q - p0_raw, v) * v

    # 4. Determine the length based on the point cloud bounds
    # Project all points of the smaller cluster to find the min/max extent
    # (Usually the Stem tube defines the weld length in a T-joint)
    pts = np.asarray(cluster2_pcd.points)
    projections = np.dot(pts - p0_anchored, v)
    line_min = np.min(projections)
    line_max = np.max(projections)

    # Define final endpoints
    start_point = p0_anchored + line_min * v
    end_point = p0_anchored + line_max * v

    return start_point, end_point

def create_weld_segment(start, end):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]]) # Red
    return line_set

def create_thick_line(start, end, radius=0.002, color=[1, 0, 0]):
    """
    Creates a cylinder mesh between two points.
    radius: Thickness in meters (0.002 = 2mm)
    """
    # 1. Calculate cylinder properties
    vec = end - start
    dist = np.linalg.norm(vec)
    
    # 2. Create the cylinder (default is along Z-axis, centered at origin)
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=dist)
    mesh_cylinder.paint_uniform_color(color)

    # 3. Rotate cylinder to match the vector 'vec'
    # Find the rotation needed to move [0, 0, 1] to 'vec'
    vec = vec / dist # normalize
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis is the cross product
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    
    if axis_len > 1e-6:
        axis = axis / axis_len
        # Angle between z_axis and vec
        angle = np.arccos(np.clip(np.dot(z_axis, vec), -1.0, 1.0))
        # Get rotation matrix (Rodrigues formula)
        R = mesh_cylinder.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh_cylinder.rotate(R, center=(0, 0, 0))

    # 4. Translate to the midpoint
    midpoint = (start + end) / 2
    mesh_cylinder.translate(midpoint)

    return mesh_cylinder
def offset_line(p_start, p_end, normal, distance):
    """
    Translates a line segment along a normal vector.
    
    Args:
        p_start (list/np.array): [x, y, z] of the start point.
        p_end (list/np.array): [x, y, z] of the end point.
        normal (list/np.array): The direction vector to move along.
        distance (float): The distance to offset.
        
    Returns:
        tuple: (new_start, new_end) as numpy arrays.
    """
    # Convert inputs to numpy arrays
    ps = np.array(p_start)
    pe = np.array(p_end)
    vec_n = np.array(normal)

    # 1. Normalize the normal vector (make it unit length)
    norm = np.linalg.norm(vec_n)
    if norm == 0:
        raise ValueError("Normal vector cannot be zero-length.")
    unit_n = vec_n / norm

    # 2. Calculate the offset vector
    offset_vec = unit_n * distance

    # 3. Apply translation to both points
    new_start = ps + offset_vec
    new_end = pe + offset_vec

    return new_start, new_end

def find_t_joint_paths(pcd, table_normal = np.array([0, 0, 1]), angle_threshold_deg = 10, offset = 0.2, visualize=False):
    # find pcd associated with base and stem
    _, pcd_base, pcd_stem = find_similar_points(pcd, table_normal, angle_threshold_deg)

    if visualize:
        o3d.visualization.draw_geometries([pcd_base])
        o3d.visualization.draw_geometries([pcd_stem])

    # find planes associated with base
    base_planes = []
    base_pcds = []
    for i in range(2):
        plane_model, inliers = pcd_base.segment_plane(distance_threshold=0.02,
                                         ransac_n=3,
                                         num_iterations=1000)
        base_planes.append(plane_model)

        base_pcds.append(copy.deepcopy(pcd_base).select_by_index(inliers))
        pcd_base = pcd_base.select_by_index(inliers, invert=True)
        

    # only keep the top plane
    print(base_planes[0])
    print(base_planes[1])
    if base_planes[0][-1] > base_planes[1][-1]:
        base_plane_top = base_planes[0]
    else:
        base_plane_top = base_planes[1]

    if visualize:
        base_pcds[0].paint_uniform_color([1, 0, 0])
        base_pcds[1].paint_uniform_color([0, 1, 0])
        mesh_1 = create_mesh_from_plane(base_planes[0])
        mesh_2 = create_mesh_from_plane(base_planes[1])
        o3d.visualization.draw_geometries([base_pcds[0], base_pcds[1], mesh_1, mesh_2])

    # find planes associated with stem    
    stem_planes = []
    stem_plane_pcds = []
    for i in range(2):
        plane_model, inliers = pcd_stem.segment_plane(distance_threshold=0.005,
                                         ransac_n=3,
                                         num_iterations=1000)
        stem_plane_pcds.append(copy.deepcopy(pcd_stem).select_by_index(inliers))
        stem_planes.append(plane_model)

        pcd_stem = pcd_stem.select_by_index(inliers, invert=True)
        
    if visualize:
        stem_plane_pcds[0].paint_uniform_color([1, 0, 0])
        stem_plane_pcds[1].paint_uniform_color([0, 1, 0])
        mesh_1 = create_mesh_from_plane(stem_planes[0])
        mesh_2 = create_mesh_from_plane(stem_planes[1])
        o3d.visualization.draw_geometries([stem_plane_pcds[0], stem_plane_pcds[1], mesh_1, mesh_2])

    # find line of intersection of the two planes
    radius = 0.005
    start_point, end_point = get_centered_intersection_line(base_plane_top, stem_planes[0], pcd_base, stem_plane_pcds[0])
    start_point, end_point = offset_line(start_point, end_point, base_plane_top[:3], offset)
    start_point, end_point = offset_line(start_point, end_point, stem_planes[0][:3], offset)
    weld_segment_1_mesh = create_thick_line(start_point, end_point, radius)
    weld_segment_1_points = [start_point, end_point]

    start_point, end_point = get_centered_intersection_line(base_plane_top, stem_planes[1], pcd_base, stem_plane_pcds[1])
    start_point, end_point = offset_line(start_point, end_point, base_plane_top[:3], offset)
    start_point, end_point = offset_line(start_point, end_point, -stem_planes[1][:3], offset)
    weld_segment_2_mesh = create_thick_line(start_point, end_point, radius)
    weld_segment_2_points = [start_point, end_point]

    return weld_segment_1_mesh, weld_segment_2_mesh, weld_segment_1_points, weld_segment_2_points


def main():
    clean_pcd = True
    process_pcd = True
    if clean_pcd:
        # get point cloud and clean it
        name = "T-join_noisy.ply"
        pcd = o3d.io.read_point_cloud(name)
        pcd = pcd.random_down_sample(sampling_ratio=0.05)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(name.split(".")[0] + "_downsampled.ply", pcd)

    if process_pcd:
        name = "T-join_noisy_downsampled.ply"
        pcd = o3d.io.read_point_cloud(name)
        # pcd = copy.deepcopy(pcd_original)
        pcd.paint_uniform_color([0, 0, 1])

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=0.1)
        pcd = pcd.select_by_index(ind)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=20))
        pcd.orient_normals_towards_camera_location(camera_location=[500, 500, 500])

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])

        weld_segment_1, weld_segment_2 = find_t_joint_paths(pcd, visualize=True)

        o3d.visualization.draw_geometries([pcd, weld_segment_1, weld_segment_2, axes])

if __name__ == "__main__":
    main()







