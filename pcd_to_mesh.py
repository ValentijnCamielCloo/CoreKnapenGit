# Import libraries
import open3d as o3d
import os
import numpy as np


def pcd_to_mesh(input_name, output_name):
    # Define the folder path where your PLY files are stored and where the filtered PLY will be saved
    input_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds'
    output_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\saved_meshes'

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Specify the PLY file name to load
    ply_file_path = os.path.join(input_folder, f"{input_name}.ply")

    # Load the point cloud from the PLY file
    point_cloud = o3d.io.read_point_cloud(ply_file_path)

    # Add a small jitter (tiny random shift) to points to avoid bad meshes with holes
    jitter_amount = 1e-8  # Small value to break the alignment, but it won't be visible in the final mesh
    points = np.asarray(point_cloud.points)
    points += np.random.normal(scale=jitter_amount, size=points.shape)
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Compute normals if PCD does not have
    if point_cloud.has_normals() == False:
        print("Estimating normals...")
        point_cloud.estimate_normals()
    else:
        print("PCD already has normals. So, skip estimating normals")

    # Orient the computed normals w.r.t to the tangent plane.
    # This step will solve the normal direction issue. If this step is skipped, there might be holes in the mesh surfaces.
    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(point_cloud, 10)

    # Estimate radius for rolling ball
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    print("Minimum neighbour distance = {:.6f}".format(np.min(distances)))
    print("Maximum neighbour distance = {:.6f}".format(np.max(distances)))
    print("Average neighbour distance = {:.6f}".format(np.mean(distances)))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                        point_cloud,
                                        o3d.utility.DoubleVector([radius, radius * 2]))

    print("PCD Detail: {}".format(point_cloud))
    print("Mesh Details: {}".format(mesh))

    # Visualize the repaired mesh after flipping
    o3d.visualization.draw_geometries([mesh])

    # Define the file name for the saved mesh
    mesh_file_name = f"{output_name}.ply"

    # Full path to save the mesh
    mesh_file_path = os.path.join(output_folder, mesh_file_name)

    # Save the mesh
    o3d.io.write_triangle_mesh(mesh_file_path, mesh)

    # Print confirmation
    print(f"Mesh saved to {mesh_file_path}")
    return mesh
