import pyvista as pv
import numpy as np
import open3d as o3d

def segment_bricks(mesh_file, pcd_file, points_per_brick):
    # Load mesh and point cloud
    composite_mesh_file = fr"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\{mesh_file}"
    point_cloud_file = fr'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\{pcd_file}'

    # Load composite mesh and point cloud
    composite_mesh = pv.read(composite_mesh_file)

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
        x_bound, y_bound = component.bounds[:2], component.bounds[2:4]

        # Initialize list for points inside current component bounds
        points_inside = []

        # Loop through each point and check if it falls within the x and y bounds
        for point in points:
            px, py = point[0], point[1]

            # Check if the point is within the x and y bounds
            if x_bound[0] < px < x_bound[1] and y_bound[0] < py < y_bound[1]:
                points_inside.append(point)

        # Classify components based on whether they contain at least 5 points
        if len(points_inside) >= points_per_brick:
            mesh_points.append(i)
        else:
            mesh_no_points.append(i)

    return mesh_points, mesh_no_points, surface_mesh

mesh_points, mesh_no_points, surface_mesh = segment_bricks("composite_meshes_09-10.ply", 'translated_pcd1_08-10.ply', 5)


# Output the number of bricks classified
bricks = len(mesh_points)
no_bricks = len(mesh_no_points)
progress = round(bricks / (bricks+no_bricks) * 100,2)
print(f"{bricks} bricks have already been built")
print(f"{no_bricks} bricks still need to be built")
print(f"Progress = {progress} %")

# Visualize the mesh and components
plotter = pv.Plotter()

mesh_vis_file = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model1_08-10.ply"
mesh_vis = pv.read(mesh_vis_file)
plotter.add_mesh(mesh_vis, color='red', opacity=0.8)

# Add mesh with sufficient points
if mesh_points:
    plotter.add_mesh(surface_mesh.extract_cells(mesh_points), color='green', edge_color='black', line_width=1, opacity=1)

plotter.show()
