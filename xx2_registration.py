import open3d as o3d
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from xx0_functions_registration import draw_registration_result_original_color  # Ensure this function is defined or imported

# Define the folder paths
scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit\scans'
output_base_folder = r'D:\TUdelftGitCore\CoreKnapenGit\ProgressPilotRegistration'

# Create a folder for the current run with date and time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(output_base_folder, f"ProgressPilotRegistration_{current_time}")
os.makedirs(output_folder, exist_ok=True)

# Get all PLY files from the scans folder
ply_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith(".ply")])

# Define multi-scale parameters for Colored ICP registration
voxel_radius = [0.04, 0.02, 0.01]
max_iter = [50, 30, 14]
current_transformation = np.identity(4)

# Define the known rotation angles in degrees for initial alignment
rotation_angles = [0, 0, 45, 90, 135, 180, 225, 270, 315]

# Function to create a rotation matrix for a given angle in degrees
def get_rotation_matrix(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_val, sin_val = np.cos(angle_radians), np.sin(angle_radians)
    rotation_matrix = np.array([[cos_val, -sin_val, 0, 0],
                                [sin_val, cos_val, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return rotation_matrix

# Function to create and return a coordinate frame (XYZ axes)
def create_coordinate_frame(size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return frame

# Initialize cumulative point cloud with the first scan
cumulative_cloud = o3d.io.read_point_cloud(os.path.join(scans_folder_path, ply_files[0]))

# Lists to store fitness and inlier RMSE values
fitness_values = []
inlier_rmse_values = []

# Load and register each scan iteratively
for i in range(1, len(ply_files)):
    source_path = os.path.join(scans_folder_path, ply_files[i])
    source = o3d.io.read_point_cloud(source_path)
    print(f"Registering cumulative point cloud to {ply_files[i]}")
    
    # Apply initial rotation based on the known angle
    initial_rotation = get_rotation_matrix(rotation_angles[i])
    rotated_source = source.transform(initial_rotation)  # Create a rotated copy for visualization
    
    # Visualize the rotated source cloud before registration
    initial_rotation_visualization = f"INITIAL ALIGNMENT: Source: {ply_files[i]}, Target: Cumulative Cloud (before registration)"
    o3d.visualization.draw_geometries([rotated_source, cumulative_cloud, create_coordinate_frame()], window_name=initial_rotation_visualization)
    
    # Apply the initial rotation to the source
    source.transform(initial_rotation)
    
    # Multi-scale registration using Colored ICP
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print(f"Scale {scale + 1} - Iterations: {iter}, Voxel size: {radius}")
        
        # Before registration, show current counts
        print(f"Before registration: Cumulative cloud has {len(cumulative_cloud.points)} points, Source has {len(source.points)} points.")
        
        # Downsample cumulative cloud and source cloud
        cumulative_down = cumulative_cloud.voxel_down_sample(radius)
        source_down = source.voxel_down_sample(radius)
        
        # Estimate normals
        cumulative_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        
        # Apply Colored ICP registration
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, cumulative_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        fitness_values.append(result_icp.fitness)  # Store fitness value
        inlier_rmse_values.append(result_icp.inlier_rmse)  # Store inlier RMSE value
        print(f"ICP Result for Scale {scale + 1}:")
        print(f"  Transformation matrix:\n{current_transformation}")
        print(f"  Fitness: {result_icp.fitness}, Inlier RMSE: {result_icp.inlier_rmse}")
    
    # Transform the source to align with the cumulative point cloud
    source.transform(current_transformation)
    
    # Visualize the registered source and cumulative cloud
    window_title_registration = f"REGISTERED: Source: {ply_files[i]}, Target: Cumulative Cloud (after registration)"
    o3d.visualization.draw_geometries([source, cumulative_cloud, create_coordinate_frame()], window_name=window_title_registration)
    
    # Merge the source into the cumulative point cloud
    cumulative_cloud += source
    cumulative_output_path = os.path.join(output_folder, f"cumulative_cloud_after_scan_{i+1}.ply")
    o3d.io.write_point_cloud(cumulative_output_path, cumulative_cloud)

# Plotting histograms for fitness and inlier RMSE
plt.figure(figsize=(12, 5))

# Fitness histogram
plt.subplot(1, 2, 1)
plt.hist(fitness_values, bins=10, color='blue', alpha=0.7)
plt.title('Fitness Values Histogram')
plt.xlabel('Fitness')
plt.ylabel('Frequency')

# Inlier RMSE histogram
plt.subplot(1, 2, 2)
plt.hist(inlier_rmse_values, bins=10, color='green', alpha=0.7)
plt.title('Inlier RMSE Histogram')
plt.xlabel('Inlier RMSE')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fitness_inlier_rmse_histograms.png"))
plt.show()

print("Registration complete. Final cumulative point cloud saved in:", output_folder)
