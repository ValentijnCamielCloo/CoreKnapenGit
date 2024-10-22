import os
import laspy
import numpy as np
import open3d as o3d
import time

# File name (assuming the file is in the same directory as this script)
las_file_path = "group3.las"

# Check if the file exists
if not os.path.exists(las_file_path):
    print(f"File not found: {las_file_path}")
    exit()

print(f"Reading LAS file: {las_file_path}")

# Read the LAS file using laspy
start_time = time.time()  # Start timing the read operation
try:
    las = laspy.read(las_file_path)
    elapsed_time = time.time() - start_time
    print("LAS file read successfully in {:.2f} seconds.".format(elapsed_time))
except Exception as e:
    print(f"Error reading LAS file: {e}")
    exit()

# Extract X, Y, Z coordinates from the point cloud
try:
    points = np.vstack((las.x, las.y, las.z)).transpose()
    print(f"Extracted {points.shape[0]} points from the LAS file.")
except Exception as e:
    print(f"Error extracting points: {e}")
    exit()

# Filter points below 3.0 meters
height_limit = 3.0
filtered_points = points[points[:, 2] <= height_limit]
print(f"Filtered down to {filtered_points.shape[0]} points below {height_limit} meters.")

# Create an open3d PointCloud object and assign points
try:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    print("PointCloud object created successfully.")
except Exception as e:
    print(f"Error creating PointCloud object: {e}")
    exit()

# Downsample the point cloud using voxel grid filtering
voxel_size = 0.1  # Adjust this value as needed
downsampled_pcd = pcd.voxel_down_sample(voxel_size)
print(f"Downsampled from {len(pcd.points)} to {len(downsampled_pcd.points)} points.")

# Outlier removal
distance_threshold = 0.1  # Distance threshold for outlier removal
if len(downsampled_pcd.points) > 0:
    cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    print(f"Removed {len(downsampled_pcd.points) - len(ind)} outliers.")
    downsampled_pcd = downsampled_pcd.select_by_index(ind)

# Visualize the downsampled and filtered point cloud
try:
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Filtered and Downsampled LAS Point Cloud", width=800, height=600)
    print("Visualization window opened.")
except Exception as e:
    print(f"Error during visualization: {e}")
