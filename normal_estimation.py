import open3d as o3d
import numpy as np
from statistics import mean

# Step 1: Read the PLY file
point_cloud1 = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\ProgressPilot\ProgressPilot_27_20241023_122338\cluster_kmeans_0.ply")
point_cloud2 = o3d.io.read_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\ProgressPilot\ProgressPilot_27_20241023_122338\cluster_kmeans_1.ply")

# point_cloud_merge = point_cloud1 + point_cloud2

import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np


def filter_normals_based_on_local_neighborhood(pcd, radius=0.1, threshold_angle=30.0, max_nn=30):
    """
    Filter points whose normals deviate significantly from their local neighborhood normals.

    :param pcd: The input point cloud.
    :param radius: Radius for local neighborhood search.
    :param threshold_angle: Angular threshold (in degrees) for filtering points based on normals.
    :param max_nn: Maximum number of neighbors to consider in the local search.
    :return: Filtered point cloud.
    """
    # Create a KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Store indices of points that meet the angular criteria
    indices_to_keep = []

    for i in range(len(points)):
        # Search for neighbors within the specified radius
        [k, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)

        if k < 3:  # If there are too few neighbors, skip this point
            continue

        # Get the normals of the neighboring points
        neighbor_normals = normals[idx[:max_nn]]  # Limit to max_nn neighbors if necessary

        # Compute the average normal of the neighbors
        local_avg_normal = np.mean(neighbor_normals, axis=0)
        local_avg_normal /= np.linalg.norm(local_avg_normal)  # Normalize the local average normal

        # Compare the current point's normal with the local average normal
        current_normal = normals[i]
        current_normal /= np.linalg.norm(current_normal)  # Normalize the current normal

        # Calculate the angle between the current normal and the local average normal
        dot_product = np.dot(current_normal, local_avg_normal)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle = np.degrees(angle)

        # Keep the point if the angle is within the threshold
        if angle <= threshold_angle:
            indices_to_keep.append(i)

    # Select the points that meet the normal consistency criterion
    filtered_pcd = pcd.select_by_index(indices_to_keep)
    return filtered_pcd


# Step 1: Load the point cloud
point_cloud_merge = o3d.io.read_point_cloud(
    r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\ProgressPilot\ProgressPilot_27_20241023_122338\cluster_DBSCAN_0.ply")

# Step 2: Voxel downsampling (if needed)
# voxel_size = 0.02  # Adjust voxel size based on your needs
# downsampled_pc = point_cloud_merge.voxel_down_sample(voxel_size=voxel_size)

# Step 3: Estimate normals on the point cloud
# point_cloud_merge.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 4: Optionally clean the point cloud based on spatial outliers
cl, ind = point_cloud_merge.remove_radius_outlier(nb_points=15, radius=0.02)
cleaned_pcd = point_cloud_merge.select_by_index(ind)

cleaned_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 5: Filter based on local normal consistency
radius = 0.1  # Define the radius for local neighborhood
threshold_angle = 30.0  # Define the angular threshold in degrees
max_nn = 30  # Maximum number of neighbors for local normal comparison

filtered_pcd = filter_normals_based_on_local_neighborhood(cleaned_pcd, radius=radius, threshold_angle=threshold_angle,
                                                          max_nn=max_nn)

# Step 6: Visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)

# Step 6: Extract normals and convert them to a list
normals = np.asarray(point_cloud_merge.normals)
meanx = mean(normals[:,0])
meany = mean(normals[:,1])
meanz = mean(normals[:,2])

# Step 7: Print or save the list of normals
# print("List of normals:")
# print(normals)

print(f"mean x: {meanx}")
print(f"mean y: {meany}")
print(f"mean z: {meanz}")
