import pyvista as pv
import numpy as np
import open3d as o3d

# Load mesh and point cloud
composite_mesh_file = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes_09-10.ply"
mesh_vis_file = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes_10-10.ply"
point_cloud_file = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds\translated_pcd1_08-10.ply'

# Load composite mesh and point cloud
composite_mesh = pv.read(composite_mesh_file)
mesh_vis = pv.read(mesh_vis_file)

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
    if len(points_inside) >= 5:
        mesh_points.append(i)
    else:
        mesh_no_points.append(i)

# Output the number of bricks classified
bricks = len(mesh_points)
no_bricks = len(mesh_no_points)
progress = round(bricks / (bricks+no_bricks) * 100,2)
print(f"{bricks} bricks have already been built")
print(f"{no_bricks} bricks still need to be built")
print(f"Progress = {progress} %")

# Visualize the mesh and components
plotter = pv.Plotter()
plotter.add_mesh(mesh_vis, color='red', opacity=0.8)

# Add mesh with sufficient points
if mesh_points:
    plotter.add_mesh(surface_mesh.extract_cells(mesh_points), color='green', edge_color='black', line_width=1, opacity=1)

# Visualise the points of the scan (optional)
# plotter.add_mesh

plotter.show()
