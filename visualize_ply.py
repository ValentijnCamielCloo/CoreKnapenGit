import open3d as o3d    #Version 0.18.0
import numpy as np      #Version 1.26.4
import copy
import pyvista as pv
import os

# Define the folder path where your PLY files are stored
folder_path = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans'

# Specify the PLY file names
ply_file_1 = os.path.join(folder_path, "Scan_46_20241015_161510_filtered.ply")
# ply_file_2 = os.path.join(folder_path, "point_cloud_20241001_132325.ply")
# ply_file_3 = os.path.join(folder_path, "point_cloud_20241001_132506.ply")
pcd = o3d.io.read_point_cloud(ply_file_1)

o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries_with_editing([pcd])
# o3d.visualization.draw_geometries_with_vertex_selection([pcd])

# Create a plotter object for visualization
plotter = pv.Plotter()

# Function to load point cloud
def load_point_cloud(ply_file):
    return pv.read(ply_file)

# Load the first point cloud
point_cloud_1 = load_point_cloud(ply_file_1)

# Load the second point cloud
# point_cloud_2 = load_point_cloud(ply_file_2)
# point_cloud_3 = load_point_cloud(ply_file_3)
# Add the first point cloud (yellow)
# plotter.add_points(point_cloud_1, rendr_points_as_spheres=True, point_size=5, color='yellow', label='Point Cloud 1')
#
# # Add the second point cloud (red)
# # plotter.add_points(point_cloud_2, render_points_as_spheres=True, point_size=5, color='red', label='Point Cloud 2')
# # plotter.add_points(point_cloud_3, render_points_as_spheres=True, point_size=5, color='green', label='Point Cloud 3')
#
# # Set the title and show the plot
# plotter.set_background("white")  # Set background color
# plotter.add_legend()  # Show legend for point clouds
# plotter.show()  # Display the plot
e