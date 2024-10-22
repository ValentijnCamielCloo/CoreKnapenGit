import open3d as o3d
import numpy as np

# Step 1: Read the point cloud from the PLY file
point_cloud = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans\Scan_35_20241011_110602_filtered.ply")
o3d.visualization.draw_geometries([point_cloud])
# Step 2: Convert points from depth-based to Cartesian system
# Extract the points as a NumPy array
points = np.asarray(point_cloud.points)

# Step 3: Swap the Y and Z axes
# New coordinate system: [X, Z (height), Y (depth)]
# CHANGE: only switch y and z
converted_points = np.zeros_like(points)
converted_points[:, 0] = points[:, 0]  # X remains the same
converted_points[:, 1] = points[:, 2]  # Z (depth) becomes Y (depth)
converted_points[:, 2] = -points[:, 1]  # Y (height) becomes Z (height)

# Step 4: Update the point cloud with the converted points
point_cloud.points = o3d.utility.Vector3dVector(converted_points)

# Step 5: Save the converted point cloud to a new file (optional)
o3d.io.write_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\converted_point_clouds\converted_pcd_14-10.ply", point_cloud)

# Step 6: Visualize the converted point cloud
o3d.visualization.draw_geometries([point_cloud])
