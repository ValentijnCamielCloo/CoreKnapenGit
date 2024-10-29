import open3d as o3d  # Version 0.18.0
import numpy as np  # Version 1.26.4
import copy

# # Function to visualize the registration 
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)

#     # Apply the transformation to the source point cloud
#     source_temp.transform(transformation)
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

#     # Use Open3D's built-in visualization to display the point clouds
#     window_title = f"Registration: Source: {source_temp}, Target: {target_temp}"
#     o3d.visualization.draw_geometries([source_temp, target_temp, coordinate_frame], window_name=window_title)

# Function for downsampling the point cloud
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = 0.01
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = 0.01
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (black): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

def apply_initial_alignment(pcd, index):
    # Rotation angles in degrees for counterclockwise rotation
    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    # Get the corresponding angle for the given index
    angle = np.radians(rotation_angles[index])  # Convert to radians

    # Create a rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation
    pcd.transform(rotation_matrix)

    return pcd

# Adjust global registration function in xx0_functions_registration.py
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    # Setting a slightly larger distance threshold may help with matching
    distance_threshold = 0.1
    print(f"Starting global registration with point-to-plane and distance threshold: {distance_threshold:.3f}")

    reg_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # Point-to-plane
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    print("Global registration successful.")
    print("Global transformation matrix:\n", reg_global.transformation)
    return reg_global
    

def apply_z_constraint(transformation):
    """
    Constrain Z-axis values in the transformation matrix.
    This removes any translation along the Z-axis and restricts Z rotation.
    """
    constrained_transformation = transformation.copy()
    constrained_transformation[2, :] = [0, 0, 1, 0]  # Set Z rotation and translation to zero
    return constrained_transformation

def execute_local_registration(source_down, target_down, voxel_size, trans_init):
    threshold = voxel_size * 0.5
    print("Apply point-to-plane ICP with Z-axis constraint for local registration")

    # Perform the local ICP registration as usual
    reg_local = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Apply Z constraint to the resulting transformation
    reg_local.transformation = apply_z_constraint(reg_local.transformation)

    print("Local registration completed.")
    print("Local transformation matrix:\n", reg_local.transformation)
    return reg_local



def evaluation_registration(source_down, target_down, threshold, result_reg):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, threshold, result_reg.transformation)
    print("Evaluation: ",evaluation)

import open3d as o3d
import numpy as np

def visualize_initial_alignment(scans, rotation_angles):
    """
    Visualizes all scans in their initial alignment within a single window.
    
    Parameters:
    - scans: List of point cloud objects representing each scan.
    - rotation_angles: List of angles (in degrees) for initial alignment, corresponding to each scan.
    
    Note: The first scan should have a 0-degree rotation.
    """
    transformed_scans = []
    
    for i, scan in enumerate(scans):
        # Convert degrees to radians for rotation
        angle_rad = np.deg2rad(rotation_angles[i])
        
        # Define the rotation matrix around the Z-axis
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply the rotation to the scan
        rotated_scan = scan.transform(rotation_matrix)
        
        # Add to list of transformed scans
        transformed_scans.append(rotated_scan)
    
    # Create a visualizer window to display all rotated scans
    o3d.visualization.draw_geometries(transformed_scans)

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

# Custom function to visualize with a title
def visualize_with_title(source, target, transformation, title):
    # Create a temporary source point cloud and apply the transformation
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    if source.has_colors():
        source_temp.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
    source_temp.transform(transformation)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(source_temp)
    vis.add_geometry(target)
    vis.run()
    vis.destroy_window()
