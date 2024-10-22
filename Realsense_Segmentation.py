import os
import re
from datetime import datetime
import open3d as o3d
import numpy as np

# Folder where scans are stored
folder_path = "output"

# Define regex pattern for scan filenames: Scan_number_date_time_filtered.ply
pattern = re.compile(r"Scan_(\d+)_(\d{8})_(\d{6})_filtered\.ply")

def get_latest_scan(folder_path):
    latest_scan = None
    latest_datetime = None

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            scan_date = match.group(2)  # e.g., '20241011'
            scan_time = match.group(3)  # e.g., '110602'
            # Combine date and time into a single datetime object
            scan_datetime = datetime.strptime(scan_date + scan_time, "%Y%m%d%H%M%S")

            # Find the most recent scan
            if latest_datetime is None or scan_datetime > latest_datetime:
                latest_datetime = scan_datetime
                latest_scan = filename

    return latest_scan

# Function to create a grid on the X-Y plane
def create_grid(size=10.0, n_lines=100):
    lines = []
    for i in range(n_lines):
        coord = -size + i * (2 * size / (n_lines - 1))
        lines.append([[-size, coord, 0], [size, coord, 0]])  # Parallel to X-axis
        lines.append([[coord, -size, 0], [coord, size, 0]])  # Parallel to Y-axis
    
    # Create line set for grid
    grid_points = np.array([point for line in lines for point in line])
    grid_lines = np.array([[2 * i, 2 * i + 1] for i in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(grid_points)
    line_set.lines = o3d.utility.Vector2iVector(grid_lines)
    line_set.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray grid color
    return line_set

# Function to create text labels at axis ends
def create_axis_labels():
    text_meshes = []
    text_data = [
        ("X", [1.2, 0, 0]),  # Label for X-axis
        ("Y", [0, 1.2, 0]),  # Label for Y-axis
        ("Z", [0, 0, 1.2]),  # Label for Z-axis
    ]
    
    for text, position in text_data:
        text_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Create a small sphere for each label
        text_mesh.translate(position)
        text_mesh.paint_uniform_color([1, 0, 0])  # Red color for labels
        text_meshes.append(text_mesh)
    
    return text_meshes

# Get the latest scan file
latest_scan_file = get_latest_scan(folder_path)
if latest_scan_file:
    print(f"Latest scan file: {latest_scan_file}")
    
    #ADDED
    folder_path = "merged"
    latest_scan_file = "Scan_48_20241015_161604_merged"
    #ADDED

    ply_path = os.path.join(folder_path, latest_scan_file)
    
    # Load the full-resolution point cloud from the latest scan
    pcd = o3d.io.read_point_cloud(ply_path)

    # Apply Y-axis filtering to isolate floor points based on real-world height (Y represents height)
    y_values = np.asarray(pcd.points)[:, 1]  # Get Y-axis values (height in real-world)
    floor_threshold = 0.1  # Adjust this based on the real-world height of the floor
    floor_indices = np.where(y_values < floor_threshold)[0]

    # Remove floor points based on Y-values
    pcd_without_floor = pcd.select_by_index(floor_indices, invert=True)
    print(f"Number of points after Y-filtered floor removal: {len(pcd_without_floor.points)}")

    # Perform improved plane segmentation using RANSAC
    plane_model, inliers = pcd_without_floor.segment_plane(distance_threshold=0.005,
                                                           ransac_n=3,
                                                           num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Further filter the floor points with RANSAC inliers
    pcd_without_floor = pcd_without_floor.select_by_index(inliers, invert=True)
    print(f"Number of points after RANSAC floor removal: {len(pcd_without_floor.points)}")

    # Compute normals for edge detection on the full-resolution point cloud
    pcd_without_floor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01, max_nn=30))

    # Optionally, orient the normals consistently
    pcd_without_floor.orient_normals_consistent_tangent_plane(k=50)

    # Adjust DBSCAN clustering (use the full-resolution cloud)
    eps_value = 0.01  # Try reducing epsilon for fine clusters
    min_points_value = 5  # Lower min points to capture smaller brick clusters

    # Perform clustering for segmentation on the full-resolution point cloud without the floor
    labels = np.array(pcd_without_floor.cluster_dbscan(eps=eps_value, 
                                                       min_points=min_points_value, 
                                                       print_progress=True))

    # Visualize segmentation using random colors
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = np.random.rand(max_label + 1, 3)  # Random colors for each cluster
    colors = np.vstack([colors, [0, 0, 0]])  # Black color for noise/unlabeled points
    pcd_colors = colors[labels] if labels.max() > -1 else [[0, 0, 0]] * len(labels)
    pcd_without_floor.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Create a grid for the X-Y plane and axis labels
    grid = create_grid(size=1.0, n_lines=20)
    axis_labels = create_axis_labels()

    # Visualize the result with grid and axis labels
    o3d.visualization.draw_geometries([pcd_without_floor, grid, *axis_labels])

else:
    print("No valid scan files found in the folder.")
