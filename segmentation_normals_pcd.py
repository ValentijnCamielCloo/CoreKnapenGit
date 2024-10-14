import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import pyvista as pv
import os
from datetime import datetime

# Step 1: Read the PLY file
point_cloud = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans\Scan_35_20241011_110602_filtered.ply")

# Step 2: Voxel downsampling (optional, adjust the voxel size if necessary)
voxel_size = 0.02  # Adjust voxel size based on your needs
downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)

# Step 3: Estimate normals on the downsampled point cloud
downsampled_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 4: Flip normals towards a specific direction (e.g., towards the camera or origin)
# This step ensures that all normals point towards the camera located at [0, 0, 0]
downsampled_pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

# Step 5: Visualize the downsampled point cloud with oriented normals
o3d.visualization.draw_geometries([downsampled_pc], point_show_normal=True)

# Optional: Save the oriented point cloud with normals
# o3d.io.write_point_cloud(r"C:\path\to\oriented_point_cloud.ply", downsampled_pc)

# Optional: If you want to access normals programmatically, you can do so like this:
normals = np.asarray(downsampled_pc.normals)

# Specify the number of clusters you expect
n_clusters = 3
clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

# Step 6: Get the cluster labels
labels = clustering.labels_

# Step 7: Separate points based on the clusters
max_label = labels.max()
clusters = []
for i in range(max_label + 1):
    indices = np.where(labels == i)[0]
    cluster = downsampled_pc.select_by_index(indices)
    clusters.append(cluster)

# Create a new unique folder based on the current date and time
timestamp = datetime.now().strftime('%m%d_%H')
output_folder = rf"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_{timestamp}"

# Step 8: Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save each cluster as a separate point cloud in the new folder
for i, cluster in enumerate(clusters):
    o3d.io.write_point_cloud(rf"{output_folder}/cluster_{i}.ply", cluster)

plotter = pv.Plotter()

# Step 9: Visualize the clusters
colors = np.random.rand(max_label + 1, 3)
for i, cluster in enumerate(clusters):
    color = colors[i]
    pcd_cluster = np.asarray(cluster.points)
    plotter.add_mesh(pcd_cluster, color=color)

plotter.show()
