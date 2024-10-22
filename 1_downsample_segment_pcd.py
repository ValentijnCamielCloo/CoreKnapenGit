import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import pyvista as pv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Step 1: Read the PLY file
point_cloud = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans\Scan_46_20241015_161510_filtered.ply")
o3d.visualization.draw_geometries([point_cloud])

# Step 3: Voxel downsampling (optional, adjust the voxel size if necessary)
voxel_size = 0.01  # Adjust voxel size based on your needs
downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)

# Step 4: Estimate normals on the downsampled point cloud
downsampled_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 5: Flip normals towards a specific direction (e.g., towards the camera or origin)
downsampled_pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

# Step 6: Visualize the downsampled point cloud with oriented normals
o3d.visualization.draw_geometries([downsampled_pc], point_show_normal=False)
o3d.visualization.draw_geometries([downsampled_pc], point_show_normal=True)

# Step 5: Access the normals
normals = np.asarray(downsampled_pc.normals)


def elbow_method(normals, max_k=10, save_path="elbow_plot.png"):
    inertia_values = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(normals)
        inertia_values.append(kmeans.inertia_)

    # Find the elbow point using KneeLocator
    knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
    optimal_k = knee_locator.elbow  # This is the elbow point

    # Plot the elbow graph
    plt.plot(k_range, inertia_values, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for Optimal k (Elbow at k={optimal_k})')
    plt.axvline(x=optimal_k, color='red', linestyle='--')
    plt.show()

    # Save the plot as an image file
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Elbow plot saved as {save_path}")

    return optimal_k


# Automatically choose the optimal number of clusters
n_clusters = elbow_method(normals, max_k=10)
print(f'amount of clusters: {n_clusters}')

# Step 8: Apply KMeans clustering using the chosen number of clusters
clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

# Step 8: Get the cluster labels
labels = clustering.labels_

# Step 9: Separate points based on the clusters
max_label = labels.max()
clusters = []
for i in range(max_label + 1):
    indices = np.where(labels == i)[0]
    cluster = downsampled_pc.select_by_index(indices)
    clusters.append(cluster)

# Define a threshold to determine if a normal is facing upwards
upward_threshold = 0.5  # Adjust based on the Z-axis being "up"

# Step 10: Select clusters that do not have normals pointing upwards
non_upward_clusters = []
for i, cluster in enumerate(clusters):
    # Calculate the mean normal of the cluster
    mean_normal = np.mean(np.asarray(cluster.normals), axis=0)
    print(f"{i}: {mean_normal}")
    if mean_normal[2] <= upward_threshold:  # Check if the Z-component is below or equal to the threshold
        non_upward_clusters.append(cluster)

# Uncomment the lines below to save the clusters to a folder if needed
timestamp = datetime.now().strftime('%m%d_%H%M')
output_folder = rf"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_{timestamp}"
os.makedirs(output_folder, exist_ok=True)
for i, cluster in enumerate(non_upward_clusters):
    o3d.io.write_point_cloud(rf"{output_folder}/non_upward_cluster_{i}.ply", cluster)

# Step 11: Visualize the non-upward clusters using PyVista
plotter = pv.Plotter()
colors = np.random.rand(len(non_upward_clusters), 3)  # Generate colors for each non-upward cluster

for i, cluster in enumerate(non_upward_clusters):
    color = colors[i]
    pcd_cluster = np.asarray(cluster.points)
    plotter.add_mesh(pcd_cluster, color=color)

plotter.show()
