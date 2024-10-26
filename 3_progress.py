import pyvista as pv
import numpy as np
import open3d as o3d

def segment_bricks(mesh_file, pcd_file, points_per_brick):
    # Load mesh and point cloud
    composite_mesh_file = fr"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\{mesh_file}"
    point_cloud_file = fr'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\{pcd_file}'

    # Load composite mesh and point cloud
    composite_mesh = pv.read(composite_mesh_file)
    normal_mesh = composite_mesh.point_data['Normals']
    mean_normal = np.mean(normal_mesh, axis=0)
    print(mean_normal)

    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)

    # Extract surface mesh and number of components
    surface_mesh = composite_mesh.extract_surface()
    n_components = surface_mesh.n_cells

    # Initialize lists for components with and without points
    mesh_points = []
    mesh_no_points = []

    # Iterate over each component in the mesh
    for i in range(n_components):
        component = surface_mesh.extract_cells([i])
        if (mean_normal[0] == 0) and (mean_normal[1] == 0):
            print('x and y')
            x_bound, y_bound = component.bounds[:2], component.bounds[2:4]
            print(f'x bound: {x_bound}, y bound: {y_bound}')
            # Initialize list for points inside current component bounds
            points_inside = []

            # Loop through each point and check if it falls within the x and y bounds
            for point in points:
                px, py = point[0], point[1]

                # Check if the point is within the x and y bounds
                if x_bound[0] < px < x_bound[1] and y_bound[0] < py < y_bound[1]:
                    points_inside.append(point)

        elif (mean_normal[0] == 0) and (mean_normal[2] == 0):
            print('x and z')
            x_bound, z_bound = component.bounds[:2], component.bounds[4:6]

            # Initialize list for points inside current component bounds
            points_inside = []

            # Loop through each point and check if it falls within the x and y bounds
            for point in points:
                px, pz = point[0], point[2]

                # Check if the point is within the x and z bounds
                if x_bound[0] < px < x_bound[1] and z_bound[0] < pz < z_bound[1]:
                    points_inside.append(point)

        elif (mean_normal[1] == 0) and (mean_normal[2] == 0):
            print('y and z')
            y_bound, z_bound = component.bounds[2:4], component.bounds[4:6]

            # Initialize list for points inside current component bounds
            points_inside = []

            # Loop through each point and check if it falls within the x and y bounds
            for point in points:
                py, pz = point[1], point[2]

                # Check if the point is within the y and z bounds
                if y_bound[0] < py < y_bound[1] and z_bound[0] < pz < z_bound[1]:
                    points_inside.append(point)

            else:
                print("something went wrong with finding the right bounds")

        # Classify components based on whether they contain at least 5 points
        if len(points_inside) >= points_per_brick:
            mesh_points.append(i)
        else:
            mesh_no_points.append(i)

    return mesh_points, mesh_no_points, surface_mesh

mesh_points1, mesh_no_points1, surface_mesh1 = segment_bricks("composite_meshes2-1_15-10.ply", 'translated_cluster_3-0_1510.ply', 1)
mesh_points2, mesh_no_points2, surface_mesh2 = segment_bricks("composite_meshes2-2_15-10.ply", 'translated_cluster_3-1_1510.ply', 1)



# Output the number of bricks classified
bricks1 = len(mesh_points1)
no_bricks1 = len(mesh_no_points1)
progress1 = round(bricks1 / (bricks1+no_bricks1) * 100,2)
print('Wall 1:')
print(f"{bricks1} bricks have already been built")
print(f"{no_bricks1} bricks still need to be built")
print(f"Progress = {progress1} %")

bricks2 = len(mesh_points2)
no_bricks2 = len(mesh_no_points2)
progress2 = round(bricks2 / (bricks2+no_bricks2) * 100,2)
print('Wall 2:')
print(f"{bricks2} bricks have already been built")
print(f"{no_bricks2} bricks still need to be built")
print(f"Progress = {progress2} %")

print(f"In total {no_bricks1+no_bricks2} bricks need to be built")

# Visualize the mesh and components
plotter = pv.Plotter()

mesh_vis_file = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model_corner2_15-10.ply"
mesh_vis = pv.read(mesh_vis_file)

# mesh_vis_file2 = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model2_15-10.ply"
# mesh_vis2 = pv.read(mesh_vis_file2)

plotter.add_mesh(mesh_vis, color='red', opacity=0.8)
# plotter.add_mesh(mesh_vis2, color='red', opacity=0.8)

# Add mesh with sufficient points
if mesh_points1:
    plotter.add_mesh(surface_mesh1.extract_cells(mesh_points1), color='green', edge_color='black', line_width=1, opacity=1)
if mesh_points2:
    plotter.add_mesh(surface_mesh2.extract_cells(mesh_points2), color='green', edge_color='black', line_width=1, opacity=1)

plotter.show()
#
# # Load mesh and point cloud
# composite_mesh_file = fr"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2_14-10.ply"
# point_cloud_file = fr'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\translated_cluster_0.ply'
#
# plotter = pv.Plotter()
# # Load composite mesh and point cloud
# composite_mesh = pv.read(composite_mesh_file)
# plotter.add_mesh(composite_mesh)
# # o3d.visualization.draw_geometries([composite_mesh])
# normal_mesh = composite_mesh.point_data['Normals']
# mean_normal = np.mean(normal_mesh, axis=0)
# print(mean_normal)
#
# pcd = o3d.io.read_point_cloud(point_cloud_file)
# # o3d.visualization.draw_geometries([pcd])
# points = np.asarray(pcd.points)
# plotter.add_mesh(points)
# plotter.show()
#
# # Extract surface mesh and number of components
# surface_mesh = composite_mesh.extract_surface()
# n_components = surface_mesh.n_cells
#
# # Initialize lists for components with and without points
# mesh_points = []
# mesh_no_points = []
#
# # Iterate over each component in the mesh
# for i in range(n_components):
#     component = surface_mesh.extract_cells([i])
#     if (mean_normal[0] == 0) and (mean_normal[1] == 0):
#         print('x and y')
#         x_bound, y_bound = component.bounds[:2], component.bounds[2:4]
#         print(f'x bound: {x_bound}, y bound: {y_bound}')
#         # Initialize list for points inside current component bounds
#         points_inside = []
#
#         # Loop through each point and check if it falls within the x and y bounds
#         for point in points:
#             px, py = point[0], point[1]
#
#             # Check if the point is within the x and y bounds
#             if x_bound[0] < px < x_bound[1] and y_bound[0] < py < y_bound[1]:
#                 points_inside.append(point)
#
#     elif (mean_normal[0] == 0) and (mean_normal[2] == 0):
#         print('x and z')
#         x_bound, z_bound = component.bounds[:2], component.bounds[4:6]
#
#         # Initialize list for points inside current component bounds
#         points_inside = []
#
#         # Loop through each point and check if it falls within the x and y bounds
#         for point in points:
#             px, pz = point[0], point[2]
#
#             # Check if the point is within the x and z bounds
#             if x_bound[0] < px < x_bound[1] and z_bound[0] < pz < z_bound[1]:
#                 points_inside.append(point)
#
#     elif (mean_normal[1] == 0) and (mean_normal[2] == 0):
#         print('y and z')
#         y_bound, z_bound = component.bounds[2:4], component.bounds[4:6]
#
#         # Initialize list for points inside current component bounds
#         points_inside = []
#
#         # Loop through each point and check if it falls within the x and y bounds
#         for point in points:
#             py, pz = point[1], point[2]
#
#             # Check if the point is within the y and z bounds
#             if y_bound[0] < py < y_bound[1] and z_bound[0] < pz < z_bound[1]:
#                 points_inside.append(point)
#
#         else:
#             print("something went wrong with finding the right bounds")
#
#     # Classify components based on whether they contain at least 5 points
#     if len(points_inside) >= 1:
#         mesh_points.append(i)
#     else:
#         mesh_no_points.append(i)
#
# # Output the number of bricks classified
# bricks = len(mesh_points)
# no_bricks = len(mesh_no_points)
# progress = round(bricks / (bricks+no_bricks) * 100,2)
# print(f"{bricks} bricks have already been built")
# print(f"{no_bricks} bricks still need to be built")
# print(f"Progress = {progress} %")
#
# # Visualize the mesh and components
# plotter = pv.Plotter()
#
# mesh_vis_file1 = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2_14-10.ply"
# mesh_vis1 = pv.read(mesh_vis_file1)
#
#
# plotter.add_mesh(mesh_vis1, color='red', opacity=0.8)
#
# # Add mesh with sufficient points
# if mesh_points:
#     plotter.add_mesh(surface_mesh.extract_cells(mesh_points), color='green', edge_color='black', line_width=1, opacity=1)
#
#
# plotter.show()