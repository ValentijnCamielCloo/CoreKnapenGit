import open3d as o3d
import numpy as np
from statistics import mean

# Step 1: Read the PLY file
point_cloud = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans\Scan_43_20241015_142545_filtered.ply")

# Step 2: Voxel downsampling
voxel_size = 0.02  # Adjust voxel size based on your needs
downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)

# Step 3: Estimate normals on the downsampled point cloud
downsampled_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


# Step 4: Flip normals towards a specific direction (e.g., towards the camera or origin)
downsampled_pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])  # Origin [0, 0, 0]

# Step 5: Visualize the downsampled point cloud with flipped normals
o3d.visualization.draw_geometries([downsampled_pc], point_show_normal=True)

# Step 6: Extract normals and convert them to a list
normals = np.asarray(downsampled_pc.normals)
meanx = mean(normals[:,0])
meany = mean(normals[:,1])
meanz = mean(normals[:,2])

# Step 7: Print or save the list of normals
# print("List of normals:")
# print(normals)

print(f"mean x: {meanx}")
print(f"mean y: {meany}")
print(f"mean z: {meanz}")
