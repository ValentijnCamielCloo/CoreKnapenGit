import open3d as o3d
import numpy as np
import pyvista as pv

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


# Load model (two walls on two sides)
mesh1 = pv.read(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2-1_15-10.ply")
mesh2 = pv.read(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2-2_15-10.ply")

# Function to compute normals if they are not present
def ensure_normals(mesh):
    if 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True)

# Ensure normals for both meshes
ensure_normals(mesh1)
ensure_normals(mesh2)

# Access vertex normals
normals1 = mesh1.point_data['Normals']
normals2 = mesh2.point_data['Normals']

# Mean normal
mean_normal1_model = np.mean(normals1, axis=0)

# # Print the normals for both meshes
# print("Normals for Mesh 1:")
# print(normals1)
#
# print("\nNormals for Mesh 2:")
# print(normals2)



# Load segmented point clouds from scan
pcd_1 = o3d.io.read_point_cloud(
    r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_1015_1714\non_upward_cluster_0.ply")
# Remove outliers (Statistical outlier removal)

# point_cloud1, ind1 = pcd_1.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
# o3d.visualization.draw_geometries([point_cloud1])

point_cloud1, ind1 = pcd_1.remove_radius_outlier(nb_points=12, radius=0.02)
o3d.visualization.draw_geometries([point_cloud1])

points_before1 = np.asarray(pcd_1.points)
points_after1 = np.asarray(point_cloud1.points)
print(f'pcd 1: before: {len(points_before1)}, after: {len(points_after1)}')

pcd_2 = o3d.io.read_point_cloud(
    r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_1015_1714\non_upward_cluster_1.ply")
# point_cloud2, ind2 = pcd_2.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
# o3d.visualization.draw_geometries([point_cloud2])

point_cloud2, ind2 = pcd_2.remove_radius_outlier(nb_points=12, radius=0.02)
o3d.visualization.draw_geometries([point_cloud2])

points_before2 = np.asarray(pcd_2.points)
points_after2 = np.asarray(point_cloud2.points)
print(f'pcd 2: before: {len(points_before2)}, after: {len(points_after2)}')

# Step 3: Get the points from both point clouds
points1 = np.asarray(point_cloud1.points)
points2 = np.asarray(point_cloud2.points)

filter_mask = (points2[:, 1] >= 0.0) & (points2[:, 1] <= 0.5)
points2 = points2[filter_mask]

plotter = pv.Plotter()
plotter.add_mesh(points1, color='lightblue')
plotter.add_mesh(points2, color='lightblue')
plotter.add_mesh(mesh1, color='lightgrey')
plotter.add_mesh(mesh2, color='lightgrey')
plotter.show()

# Step 5: Calculate normals for each point cloud (if they are not already calculated)
point_cloud1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
point_cloud2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 6: Calculate the mean normal for both clusters
normals1 = np.asarray(point_cloud1.normals)
normals2 = np.asarray(point_cloud2.normals)

mean_normal1 = np.mean(normals1, axis=0)
mean_normal2 = np.mean(normals2, axis=0)

# Compute the rotation matrix that rotates vec1 to vec2
rotation_matrix = rotation_matrix_from_vectors(mean_normal1, mean_normal1_model)

print(f"Rotation Matrix:\n {rotation_matrix}")

# Apply the rotation to both point clouds
rotated_points1 = points1 @ rotation_matrix.T  # Rotate points
rotated_normals1 = normals1 @ rotation_matrix.T  # Rotate normals

rotated_points2 = points2 @ rotation_matrix.T  # Rotate points
rotated_normals2 = normals2 @ rotation_matrix.T  # Rotate normals

# Update the point cloud with rotated points and normals
point_cloud1.points = o3d.utility.Vector3dVector(rotated_points1)
point_cloud1.normals = o3d.utility.Vector3dVector(rotated_normals1)

point_cloud2.points = o3d.utility.Vector3dVector(rotated_points2)
point_cloud2.normals = o3d.utility.Vector3dVector(rotated_normals2)

# Step 11: Visualize the rotated point clouds together
o3d.visualization.draw_geometries([point_cloud1, point_cloud2], point_show_normal=True)


while True:
    # Create a plotter
    plotter = pv.Plotter()

    # Add the first mesh and normals (MODEL)
    plotter.add_mesh(mesh1, color='lightgrey', label='Mesh 1')
    # plotter.add_arrows(mesh1.points, normals1 * 0.1, color='red')

    # Add the second mesh and normals (MODEL)
    plotter.add_mesh(mesh2, color='lightblue', label='Mesh 2')
    # plotter.add_arrows(mesh2.points, normals2 * 0.1, color='green')

    points_scan_1 = np.asarray(rotated_points1)
    plotter.add_mesh(points_scan_1)

    points_scan_2 = np.asarray(rotated_points2)
    plotter.add_mesh(points_scan_2)

    # Show the plot
    plotter.show()

    rot_feedback = input("Is the rotation correct?")
    if rot_feedback == 'yes' or rot_feedback == 'Yes':
        break
    else:
        rot_adjustment = input("How does it need to be rotated? [90 left, 90 right, 180]")
        if rot_adjustment == '90 left':
            # Anti-clockwise rotation (90 degrees around Z-axis)
            rotation_matrix_new = np.array([[0, -1, 0],
                                        [1, 0, 0],
                                        [0, 0, 1]])
        elif rot_adjustment == '90 right':
            # Clockwise rotation (90 degrees around Z-axis)
            rotation_matrix_new = np.array([[0, 1, 0],
                                        [-1, 0, 0],
                                        [0, 0, 1]])
        elif rot_adjustment == '180':
            # 180-degree rotation around Z-axis
            rotation_matrix_new = np.array([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]])
        else:
            raise ValueError("Direction must be '90 left', '90 right', or '180'.")

        rotated_points1 = rotated_points1 @ rotation_matrix_new.T  # Rotate points
        rotated_normals1 = rotated_normals1 @ rotation_matrix_new.T  # Rotate normals

        rotated_points2 = rotated_points2 @ rotation_matrix_new.T  # Rotate points
        rotated_normals2 = rotated_normals2 @ rotation_matrix_new.T  # Rotate normals

        # Update the point cloud with rotated points and normals
        point_cloud1.points = o3d.utility.Vector3dVector(rotated_points1)
        point_cloud1.normals = o3d.utility.Vector3dVector(rotated_normals1)

        point_cloud2.points = o3d.utility.Vector3dVector(rotated_points2)
        point_cloud2.normals = o3d.utility.Vector3dVector(rotated_normals2)

    # Step 12: Save the rotated point clouds
    o3d.io.write_point_cloud(
        r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_1015_1714\rotated_cluster_0.ply",
        point_cloud1)
    o3d.io.write_point_cloud(
        r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\segmented_point_clouds\segmentation_1015_1714\rotated_cluster_1.ply",
        point_cloud2)


# Find minimum corner point
min_x_value = min(np.min(rotated_points1[:, 0]), np.min(rotated_points2[:, 0]))
min_y_value = min(np.min(rotated_points1[:, 1]), np.min(rotated_points2[:, 1]))
min_z_value = min(np.min(rotated_points1[:, 2]), np.min(rotated_points2[:, 2]))

# New corner point
corner_point = np.array([min_x_value, min_y_value, min_z_value])

# Translate both point clouds so the new corner point is at the origin
translation_vector = -corner_point

translated_points1 = rotated_points1 + translation_vector
translated_points2 = rotated_points2 + translation_vector

# Update the point clouds with the translated points
point_cloud1.points = o3d.utility.Vector3dVector(translated_points1)
point_cloud2.points = o3d.utility.Vector3dVector(translated_points2)

point_cloud1_points = np.asarray(point_cloud1.points)
point_cloud2_points = np.asarray(point_cloud2.points)

plotter = pv.Plotter()
plotter.add_mesh(point_cloud1_points, color='lightblue')
plotter.add_mesh(point_cloud2_points, color='lightblue')
plotter.add_mesh(mesh1, color='lightgrey')
plotter.add_mesh(mesh2, color='lightgrey')
plotter.show()

# Final visualization after translation
o3d.visualization.draw_geometries([point_cloud1, point_cloud2], point_show_normal=True)

# Save translated point clouds
o3d.io.write_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\translated_cluster_3-0_1510.ply", point_cloud1)
o3d.io.write_point_cloud(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\translated_cluster_3-1_1510.ply", point_cloud2)