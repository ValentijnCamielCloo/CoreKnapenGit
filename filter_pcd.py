import numpy as np
import open3d as o3d
import os


import os
import open3d as o3d
import numpy as np

def filter_pcd(input_name, output_name, dist_camera, voxel_size=0.02, nb_neighbors=20, std_ratio=0.2):
    # Define the folder path where your PLY files are stored and where the filtered PLY will be saved
    input_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans'
    output_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\filtered_point_clouds'

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Specify the PLY file name to load
    ply_file_path = os.path.join(input_folder, f"{input_name}.ply")

    # Step 1: Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # Visualize the original point cloud
    o3d.visualization.draw_geometries([pcd])

    # Convert the points and colors of the point cloud to NumPy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Step 2: Remove points where z = 0 (and remove corresponding colors)
    non_zero_mask = points[:, 2] != 0  # Mask for points with z != 0
    non_zero_points = points[non_zero_mask]
    non_zero_colors = colors[non_zero_mask]

    # Define thresholds, invert if points are negative
    if non_zero_points[0, 2] < 0:
        lowest_threshold = - (dist_camera + 0.04)
        highest_threshold = - (dist_camera - 0.04)
    else:
        lowest_threshold = dist_camera - 0.04
        highest_threshold = dist_camera + 0.04

    # Step 3: Filter points and colors based on z-value (between lowest_threshold and highest_threshold)
    filter_mask = (non_zero_points[:, 2] >= lowest_threshold) & (non_zero_points[:, 2] <= highest_threshold)
    filtered_points = non_zero_points[filter_mask]
    filtered_colors = non_zero_colors[filter_mask]  # Apply the same mask to the colors

    print(f"Number of points before filtering: {len(points)}")
    print(f"Number of points after filtering: {len(filtered_points)}")

    # Step 4: Create a new point cloud from the filtered points and assign the filtered colors
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # # Visualize the filtered point cloud with colors
    o3d.visualization.draw_geometries([filtered_pcd])

    # Step 5: Perform Voxel Downsampling
    downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size)

    print(f"Number of points after downsampling: {len(downsampled_pcd.points)}")

    # Step 6: Remove outliers using Statistical Outlier Removal (number of neighbors and standard deviation multiplier)
    cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Select inliers (non-outliers)
    inlier_pcd = downsampled_pcd.select_by_index(ind)

    print(f"Number of points after outlier removal: {len(inlier_pcd.points)}")

    # Step 7: Visualize the filtered point cloud with outlier removal
    o3d.visualization.draw_geometries([inlier_pcd])

    # Step 8: Save the filtered point cloud to a new PLY file
    output_file = os.path.join(output_folder, f"{output_name}.ply")
    o3d.io.write_point_cloud(output_file, inlier_pcd)

    print(f"Filtered point cloud saved to: {output_file}")

# filter_pcd('Scan_31_20241011_105409_filtered', 'filtered_pcd1_11-10', 0.37,0.005)
# #
# #
# input_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans'
# ply_file_path = os.path.join(input_folder, "Scan_31_20241011_105409_filtered.ply")
# pcd = o3d.io.read_point_cloud(ply_file_path)
# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)
# # o3d.visualization.draw_geometries([pcd])