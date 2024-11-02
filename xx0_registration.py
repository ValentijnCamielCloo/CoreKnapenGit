import numpy as np
import pandas as pd
import open3d as o3d
import os
from datetime import datetime

# Define folder paths and file paths
scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit\Colored'
calculated_csv_path = r'D:\TUdelftGitCore\CoreKnapenGit\Calculated_Path_Coordinates.csv'
registered_csv_path = r'D:\TUdelftGitCore\CoreKnapenGit\registered_path_coordinates.csv'
transformed_folder_path = os.path.join(scans_folder_path, 'Transformed')

# Create a new folder for transformed files if it doesn't exist
os.makedirs(transformed_folder_path, exist_ok=True)

# Read the calculated and registered coordinates from CSV
calculated_coordinates = pd.read_csv(calculated_csv_path)
registered_coordinates = pd.read_csv(registered_csv_path)

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

    # Get the calculated coordinates for the scan
    x_calc, y_calc, z_calc = calculated_coordinates.loc[i, ['x', 'y', 'z']]
    base_rotation_calc = calculated_coordinates.loc[i, 'rotation']
    
    # Get the registered transformation for the scan
    if i > 0:  # Skip for the first scan
        reg_scan_file = scan_files[i - 1]  # Previous scan file
        reg_row = registered_coordinates[registered_coordinates['Scan_source'] == reg_scan_file]
        if not reg_row.empty:
            x_reg, y_reg, z_reg = reg_row[['X', 'Y', 'Z']].values[0]
            rot_x, rot_y, rot_z = reg_row[['Rotation_X', 'Rotation_Y', 'Rotation_Z']].values[0]
        else:
            x_reg, y_reg, z_reg = 0, 0, 0
            rot_x, rot_y, rot_z = 0, 0, 0
    else:
        x_reg, y_reg, z_reg = 0, 0, 0
        rot_x, rot_y, rot_z = 0, 0, 0

    # Calculate the final translation and rotation
    final_x = x_calc + x_reg
    final_y = y_calc + y_reg
    final_z = z_calc + z_reg
    final_rotation = base_rotation_calc + np.radians([rot_x, rot_y, rot_z])  # Adjust as needed

    # Translate to the final coordinates
    scan.translate((final_x, final_y, final_z))
    
    # Apply final rotation
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(final_rotation)
    scan.rotate(rotation_matrix, center=(final_x, final_y, final_z))

    # Save the transformed scan to the new folder
    transformed_scan_filename = f"Scan_{i + 1}_{current_time}_Transformed.ply"  # Start numbering from Scan 1
    transformed_scan_path = os.path.join(transformed_folder_path, transformed_scan_filename)
    o3d.io.write_point_cloud(transformed_scan_path, scan)

    # Visualize coordinate frame for each scan
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_frame.translate((final_x, final_y, final_z))
    coord_frame.rotate(rotation_matrix, center=(final_x, final_y, final_z))  # Apply rotation to coordinate frame as well

    # Append scan and coordinate frame to the geometries list
    geometries.extend([scan, coord_frame])

    # Draw geometries to visualize the incremental build-up
    o3d.visualization.draw_geometries(geometries, window_name=f"Transformed Scans Visualization - Up to Scan {i + 1}")

print(f"Transformed scans saved in: {transformed_folder_path}")
