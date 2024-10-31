import numpy as np
import pandas as pd
import open3d as o3d
import os
from datetime import datetime

# Define folder paths and file paths
scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit\Colored'
output_csv_path = r'D:\TUdelftGitCore\CoreKnapenGit\Calculated_Path_Coordinates.csv'
Transformed_scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit'
transformed_folder_path = os.path.join(Transformed_scans_folder_path, 'Transformed')

# Create a new folder for transformed files if it doesn't exist
os.makedirs(transformed_folder_path, exist_ok=True)

# Read the calculated coordinates from CSV
path_coordinates = pd.read_csv(output_csv_path)

# Initialize scan list and other parameters
scan_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith('.ply')])

# Initialize geometries list to visualize all scans in one go
geometries = []

# Get the current date and time for naming
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def remove_black_points(scan):
    """Remove black points (0, 0, 0) from the point cloud."""
    points = np.asarray(scan.points)
    colors = np.asarray(scan.colors)

    # Create a mask to filter out black points
    mask = ~(np.all(colors == [0, 0, 0], axis=1))
    
    # Apply the mask to points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new point cloud with the filtered points and colors
    filtered_scan = o3d.geometry.PointCloud()
    filtered_scan.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_scan.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_scan

# Loop through each scan, set its position, and apply rotation
for i, scan_file in enumerate(scan_files):
    scan_path = os.path.join(scans_folder_path, scan_file)
    scan = o3d.io.read_point_cloud(scan_path)

    # Remove black points from the scan
    scan = remove_black_points(scan)

    # Get the coordinates and rotation for each scan from CSV
    x, y, z = path_coordinates.loc[i, ['x', 'y', 'z']]
    base_rotation = path_coordinates.loc[i, 'rotation']
    
    # Translate to the calculated coordinates
    scan.translate((x, y, z))
    
    # Apply an additional rotation for each scan based on the step
    additional_rotation_angle = np.pi / 4 * i  # 45 degrees per scan
    R_additional = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, additional_rotation_angle])
    scan.rotate(R_additional, center=(x, y, z))

    # Save the transformed scan to the new folder
    transformed_scan_filename = f"Scan_{i + 2}_{current_time}_Transformed.ply"  # Start numbering from Scan 2
    transformed_scan_path = os.path.join(transformed_folder_path, transformed_scan_filename)
    o3d.io.write_point_cloud(transformed_scan_path, scan)

    # Visualize coordinate frame for each scan
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_frame.translate((x, y, z))
    coord_frame.rotate(R_additional, center=(x, y, z))  # Apply additional rotation to coordinate frame as well

    # Append scan and coordinate frame to the geometries list
    geometries.extend([scan, coord_frame])

    # Draw geometries to visualize the incremental build-up
    o3d.visualization.draw_geometries(geometries, window_name=f"Transformed Scans Visualization - Up to Scan {i+2}")

print(f"Transformed scans saved in: {transformed_folder_path}")
