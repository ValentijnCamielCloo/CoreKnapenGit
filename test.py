import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import pyvista as pv
import os
from datetime import datetime

# mesh = pv.read(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model_corner2_15-10.ply")
# # Function to compute normals if they are not present
#
# def ensure_normals(m):
#     if 'Normals' not in m.point_data:
#         m.compute_normals(inplace=True)
#
# # Ensure normals for both meshes
# ensure_normals(mesh)
#
# # Access vertex normals
# normals = mesh.point_data['Normals']
#
# # Create a visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # Add the mesh and normal lines to the visualizer
# # vis.add_geometry(mesh)
# vis.add_geometry(normals)
#
# # Set the render options
# opt = vis.get_render_option()
# opt.show_coordinate_frame = True  # Shows a coordinate frame
# opt.background_color = np.array([1, 1, 1])  # Set background to black
# opt.point_size = 5
# opt.line_width = 3
#
# # Run the visualizer
# vis.run()
#
# # Destroy the window after use
# vis.destroy_window()

# # Load the mesh
# mesh = o3d.io.read_triangle_mesh(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model_corner2_15-10.ply")
#
# # Compute the normals if they are not already present
# if not mesh.has_vertex_normals():
#     mesh.compute_vertex_normals()
#
# # Visualize the mesh with normals
# o3d.visualization.draw_geometries([mesh],
#                                   point_show_normal=True,  # Show normals
#                                   mesh_show_back_face=True)  # Display back faces too if needed

# # Load the mesh
# mesh = o3d.io.read_triangle_mesh(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model_corner2_15-10.ply")
#
# # Check if the mesh has normals, if not, compute them
# if not mesh.has_vertex_normals():
#     mesh.compute_vertex_normals()
#
# # Paint the mesh a light color for better contrast
# mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light grey color
#
# # Visualize the normals
# # Convert the normals to lines (each normal is a line segment)
# line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
# line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(line_set.points))  # Red color for normals
#
# # Create a visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # Add the mesh and normal lines to the visualizer
# # vis.add_geometry(mesh)
# vis.add_geometry(line_set)
#
# # Set the render options
# opt = vis.get_render_option()
# opt.show_coordinate_frame = True  # Shows a coordinate frame
# opt.background_color = np.array([1, 1, 1])  # Set background to black
# opt.point_size = 5
# opt.line_width = 3
#
# # Run the visualizer
# vis.run()
#
# # Destroy the window after use
# vis.destroy_window()


# Load the mesh
mesh = o3d.io.read_triangle_mesh(r"C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model\visualizing_model_corner2_15-10.ply")

# Compute vertex normals if they are not present
mesh.compute_vertex_normals()

# Paint the mesh light grey for better contrast
mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light grey color

# Create a list of lines (arrows) to represent the normals
lines = []
points = []
line_colors = []

# Scale factor for visualizing normals
scale = 0.05  # Adjust this value as needed for arrow length

# Access the vertices and normals
vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)

# Add points and corresponding arrows for each normal
for i in range(len(vertices)):
    start_point = vertices[i]  # Vertex position
    end_point = vertices[i] + normals[i] * scale  # End point of normal vector

    points.append(start_point)
    points.append(end_point)

    lines.append([2 * i, 2 * i + 1])  # Create line between two consecutive points

    # Color all normals in red
    line_colors.append([1, 0, 0])

# Create a LineSet geometry for normals
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(line_colors)

# Visualize the mesh and the normals
o3d.visualization.draw_geometries([mesh, line_set],
                                  zoom=0.8,
                                  front=[0.0, 0.0, -1.0],
                                  lookat=[0, 0, 0],
                                  up=[0, 1, 0],
                                  point_show_normal=True)
