import open3d as o3d
import numpy as np
import os
import csv

# Define file paths for source and target
source_path = r'D:\TUdelftGitCore\CoreKnapenGit\ProgressPilotRegistration\ProgressPilotRegistration_20241031_032413\Registered_4.ply'
target_path = r'D:\TUdelftGitCore\CoreKnapenGit\Scan_2_20241031_021351_Transformed.ply'

# Define output folder
output_folder = r'D:\TUdelftGitCore\CoreKnapenGit\RegistrationResult'
os.makedirs(output_folder, exist_ok=True)

# Load source and target point clouds
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)
print(f"Loaded source: {source_path}")
print(f"Loaded target: {target_path}")

# Visualize the initial alignment
initial_visualization = "INITIAL ALIGNMENT: Source (Registered_4) and Target (Scan_2)"
o3d.visualization.draw_geometries([source, target], window_name=initial_visualization)

# Define multi-scale parameters for Colored ICP registration
voxel_radius = [0.01, 0.005, 0.005]
max_iter = [300, 100, 50]
current_transformation = np.identity(4)

# Multi-scale registration using Colored ICP
for scale in range(3):
    iter = max_iter[scale]
    radius = voxel_radius[scale]
    print(f"Scale {scale + 1} - Iterations: {iter}, Voxel size: {radius}")

    # Downsample source and target clouds
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    # Estimate normals
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))

    # Apply Colored ICP registration
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=iter))
    current_transformation = result_icp.transformation
    print(f"ICP Result for Scale {scale + 1}:")
    print(f"  Transformation matrix:\n{current_transformation}")
    print(f"  Fitness: {result_icp.fitness}, Inlier RMSE: {result_icp.inlier_rmse}")

    # Visualize after each scale
    scale_visualization = f"SCALE {scale + 1}: Source and Target (after scale {scale + 1})"
    transformed_source = source_down.transform(current_transformation)
    o3d.visualization.draw_geometries([transformed_source, target_down], window_name=scale_visualization)

# Final transformation of the source to align with the target
source.transform(current_transformation)

# Visualize the final registered result
final_visualization = "FINAL REGISTRATION: Source (Registered_4) aligned with Target (Scan_2)"
o3d.visualization.draw_geometries([source, target], window_name=final_visualization)

# Export the final registered source point cloud
registered_output_path = os.path.join(output_folder, "Registered_4_aligned_to_Scan_2.ply")
o3d.io.write_point_cloud(registered_output_path, source)
print(f"Exported registered point cloud to {registered_output_path}")

# Calculate final translation and rotation
translation = current_transformation[:3, 3]
rotation = (
    np.arctan2(current_transformation[1, 0], current_transformation[0, 0]),
    np.arctan2(-current_transformation[2, 0], 
               np.sqrt(current_transformation[2, 1]**2 + current_transformation[2, 2]**2)),
    np.arctan2(current_transformation[2, 1], current_transformation[2, 2])
)

# Convert rotation from radians to degrees
rotation_degrees = tuple(np.degrees(rot) for rot in rotation)

# Print and save the final translation and rotation
print(f"Final Translation: {translation}")
print(f"Final Rotation (degrees): {rotation_degrees}")

# Write the transformation data to CSV
csv_file_path = os.path.join(output_folder, "registration_transformation.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Source", "Target", "X", "Y", "Z", "Rotation_X", "Rotation_Y", "Rotation_Z"])
    csv_writer.writerow(["Registered_4.ply", "Scan_2_20241031_021351_Transformed.ply",
                         translation[0], translation[1], translation[2], *rotation_degrees])

print(f"Transformation details saved to {csv_file_path}")
