import pyvista as pv
import numpy as np

def visualize_mesh_normals(mesh_file_paths):
    """
    Load multiple PLY files, estimate normals if not present, and visualize each mesh and its normals.

    :param mesh_file_paths: List of paths to PLY files.
    """
    # Create a PyVista Plotter
    plotter = pv.Plotter()

    for ply_file_path in mesh_file_paths:
        # Step 1: Read the PLY file
        mesh = pv.read(ply_file_path)

        # Step 2: Check if the mesh has normals
        if mesh.n_points > 0 and 'Normals' not in mesh.point_data.keys():
            print(f"Estimating vertex normals for {ply_file_path}.")
            mesh.compute_normals(inplace=True)

        # Step 3: Extract points and normals
        points = mesh.points
        normals = mesh.point_data['Normals']

        # Step 4: Add the mesh to the plotter
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, label=f'Mesh: {ply_file_path}')

        # Step 5: Create and add normals as cones
        for point, normal in zip(points, normals):
            # Create a cone at each point along the normal
            cone = pv.Cone(center=point, direction=normal, radius=0.01, height=0.1)  # Adjust radius and height for visibility
            plotter.add_mesh(cone, color='red')

    # Step 7: Show the plot
    plotter.show()

if __name__ == "__main__":
    # Replace with the paths to your PLY files
    mesh_file_paths = [
        r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2-1_15-10.ply",
        r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model\composite_meshes2-2_15-10.ply"
    ]
    visualize_mesh_normals(mesh_file_paths)
