import open3d as o3d
import numpy as np
import os
import datetime
from xx0_functions_registration import *
from xx0_functions_registration import draw_registration_result_original_color
\
# Define the folder paths 
scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit\transformed'
output_base_folder = r'D:\TUdelftGitCore\CoreKnapenGit\ProgressPilotRegistration'

# Create a folder for the current run with date and time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(output_base_folder, f"ProgressPilotRegistration_{current_time}")
os.makedirs(output_folder, exist_ok=True)

# Get all PLY files from the scans folder
ply_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith(".ply")])

# Start by loading the first scan as the initial "target" (basis)
target_file = ply_files[0]
target_path = os.path.join(scans_folder_path, target_file)
target_pcd = o3d.io.read_point_cloud(target_path)

# Loop over remaining scans and register each to the previous registered result
for i in range(1, len(ply_files)):
    source_file = ply_files[i]
    source_path = os.path.join(scans_folder_path, source_file)

    print(f"Processing: Source: {source_file}, Target: {target_file}")

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    


    # Read the next source point cloud
    source_pcd = o3d.io.read_point_cloud(source_path)
    # Visualize NON ROTATED source and target
    window_title = f"NON ROTATED: Source: {source_file}, Target: {target_file}"
    o3d.visualization.draw_geometries([source_pcd, target_pcd, coordinate_frame], window_name=window_title)

    # Apply initial alignment (rotation)
    source_pcd = apply_initial_alignment(source_pcd, i)
    # Visualize ROTATED source and target
    window_title = f"ROTATED: Source: {source_file}, Target: {target_file}"
    o3d.visualization.draw_geometries([source_pcd, target_pcd, coordinate_frame], window_name=window_title)

    # Downsample point clouds
    voxel_size = 0.01
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh  = preprocess_point_cloud(target_pcd, voxel_size)
    window_title = f"DOWNSAMPLED: Source: {source_down}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_down, target_down, coordinate_frame], window_name=window_title)
    
    # Outlier removal
    print("Removing outliers for source...")
    _, ind_source = source_down.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)
    source_down = source_down.select_by_index(ind_source)
    print("Removing outliers for target...")
    _, ind_target = target_down.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)
    target_down = target_down.select_by_index(ind_target)
    # Visualize OUTLIERS REMOVED source and target
    window_title = f"OUTLIERS REMOVED: Source: {source_down}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_down, target_down, coordinate_frame], window_name=window_title)


    # Perform global registration to obtain initial alignment
    global_trans = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print('globaltrans print:', global_trans)
    # Visualize the registration result for global registration
    source_temp = source_down.transform(global_trans.transformation)
    window_title = f"Global Registration: Source: {source_temp}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_temp, target_down, coordinate_frame], window_name=window_title)
    window_title = f"Global Registration: Source: {source_down}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_down, target_down, coordinate_frame], window_name=window_title)







    # Apply global transformation to the source cloud
    source_pcd.transform(global_trans.transformation)
    window_title = f"Global Registration: Source: {source_pcd}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_pcd, target_down, coordinate_frame], window_name=window_title)

    # Perform local ICP registration to refine alignment
    trans_local = execute_local_registration(source_down, target_down, voxel_size, global_trans.transformation)

    # Visualize the registration result for local registration
    source_temp = source_down.transform(trans_local.transformation)
    window_title = f"Local Registration: Source: {source_temp}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_temp, target_down, coordinate_frame], window_name=window_title)

    # Transform the source point cloud based on the local registration result
    source_pcd.transform(trans_local.transformation)
    window_title = f"Local Registration: Source: {source_temp}, Target: {target_down}"
    o3d.visualization.draw_geometries([source_down, target_down, coordinate_frame], window_name=window_title)

    # Save the transformed point cloud (registered result) for the next iteration
    registered_file_name = f"registered_{i}.ply"
    registered_file_path = os.path.join(output_folder, registered_file_name)
    o3d.io.write_point_cloud(registered_file_path, source_pcd)
    print(f"Registered file saved: {registered_file_path}")

    # Set the registered point cloud as the new target for the next iteration
    target_pcd += source_pcd

print("Registration process completed for all scans.")
