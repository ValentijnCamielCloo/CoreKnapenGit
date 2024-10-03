import numpy as np
import open3d as o3d
import os
from functions_registration import *
import copy

# Define the folder path where your PLY files are stored
folder_path = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\out_01-10'

# Specify the PLY file names
path = os.path.join(folder_path, "point_cloud_20241001_132244.ply")

# Read the point cloud with Open3D
pcd = o3d.io.read_point_cloud(path)

# Outlier removal
print("Statistical outlier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)

# Select inliers only
inlier_pcd = pcd.select_by_index(ind)

# Downsample the point cloud with Voxel Downsampling
voxel_size = 0.01
pcd_down, source_fpfh = preprocess_point_cloud(inlier_pcd, voxel_size)

# Make a deep copy of the point cloud for 2D conversion
pcd_2d = copy.deepcopy(pcd_down)

# Convert the points of the point cloud to a NumPy array
point = np.asarray(pcd_2d.points)

# Set the z-values to 0 (flatten onto xy-plane)
point[:, 2] = 0

# Convert the modified NumPy array back into a point cloud object
pcd_2d.points = o3d.utility.Vector3dVector(point)

# Print the modified points
print("Points converted to 2D: ", np.asarray(pcd_2d.points))

# Optionally visualize the result
o3d.visualization.draw_geometries([pcd_2d])

# Define the folder path where the translated PLY file will be saved
output_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Step 1: Convert the points of the point cloud to a NumPy array
points = np.asarray(pcd_2d.points)

# Step 2: Find the point with the lowest (x + y) value
min_xy_index = np.argmin(np.sum(points[:, :2], axis=1))
min_xy_value = points[min_xy_index]

print(f"Point with lowest (x + y): {min_xy_value}")

# Step 3: Translate the point cloud so that this point is at (0, 0)
# Subtract the x and y values of the min_xy_value from all points
points[:, 0] -= min_xy_value[0]  # Translate x values
points[:, 1] -= min_xy_value[1]  # Translate y values

# Note: z values remain unchanged

# Step 4: Update the point cloud with the translated points
pcd_2d.points = o3d.utility.Vector3dVector(points)

# Step 5: Visualize the translated point cloud (optional)
o3d.visualization.draw_geometries([pcd_2d])

# Step 6: Save the translated point cloud to a new PLY file
output_file = os.path.join(output_folder, "translated_point_cloud.ply")
o3d.io.write_point_cloud(output_file, pcd_2d)

print(f"Translation complete. The point cloud has been saved to: {output_file}")