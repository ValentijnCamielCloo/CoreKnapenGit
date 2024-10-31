import open3d as o3d
import numpy as np

# Define the file path for the point cloud
file_path = r"D:\TUdelftGitCore\CoreKnapenGit\Registered_4_aligned_to_Scan_2.ply"

# Load the point cloud
point_cloud = o3d.io.read_point_cloud(file_path)

# Define grid parameters
grid_spacing = 0.5  # grid spacing in meters (50 cm)
grid_size = 10      # total grid size in meters (extend 5 meters in each direction from origin)

# Generate grid lines along x and y at z = 0
lines = []
points = []
for i in range(-int(grid_size / 2), int(grid_size / 2) + 1):
    # Horizontal lines along the x-axis
    points.append([i * grid_spacing, -grid_size / 2, 0])
    points.append([i * grid_spacing, grid_size / 2, 0])
    lines.append([len(points) - 2, len(points) - 1])

    # Vertical lines along the y-axis
    points.append([-grid_size / 2, i * grid_spacing, 0])
    points.append([grid_size / 2, i * grid_spacing, 0])
    lines.append([len(points) - 2, len(points) - 1])

# Create line set for the grid
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0.7, 0.7, 0.7])  # light gray color for the grid

# Visualize the point cloud with the grid
o3d.visualization.draw_geometries([point_cloud, line_set], window_name="Point Cloud with Grid")
