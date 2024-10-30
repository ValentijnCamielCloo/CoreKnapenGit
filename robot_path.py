import os
import datetime
import numpy as np
import open3d as o3d
import pandas as pd

# # Define path to CSV file
# csv_path = 'robot\20241030_path_transformations.csv'
#
# # Load transformations from CSV
# transformations = pd.read_csv(csv_path)
#
# # Initial cumulative position and rotation
# cumulative_position = np.array([0, 0, 0], dtype=np.float64)
# cumulative_rotation = 0  # radians
#
# print("Calculating Rotated Translation Vectors:")
#
# # Loop through each row in the CSV to compute rotated translation vectors
# for i, row in transformations.iterrows():
#     # Extract translation and rotation from the current row
#     translation_vector = np.array([row['translation_x'], row['translation_y'], row['translation_z']])
#     rotation = row['rotation']  # in radians
#
#     # Compute rotation matrix based on the cumulative rotation around Z-axis
#     R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, cumulative_rotation])
#     rotated_translation_vector = np.dot(R, translation_vector.reshape(-1, 1)).flatten()
#
#     # Update cumulative position and rotation for next step
#     cumulative_position += rotated_translation_vector
#     cumulative_rotation += rotation
#
#     # Print the rotated translation vector and cumulative position
#     print(f"Step {i + 1}:")
#     print(f"Original translation vector: {translation_vector}")
#     print(f"Rotated translation vector: {rotated_translation_vector}")
#     print(f"Cumulative position: {cumulative_position}")
#     print(f"Rotation (radians): {cumulative_rotation}\n")


# Define the folder paths
scans_folder_path = 'scans'
output_folder = 'robot'
csv_file_path = '20241030_robot_path.csv'

# # Create a folder for the current run with date and time
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_folder = os.path.join(output_base_folder, f"ProgressPilotRegistration_{current_time}")
os.makedirs(output_folder, exist_ok=True)

# Get all PLY files from the scans folder
ply_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith(".ply")])

# Load Scan 2
scan2_file = os.path.join(scans_folder_path, ply_files[0])  # Assuming first file is Scan 2
scan2 = o3d.io.read_point_cloud(scan2_file)

# Read translations and rotations from the CSV file
transformations = pd.read_csv(csv_file_path)

# Initialize the path with the starting position
current_position = np.array([0, 0, 0], dtype=np.float64)
path_points = [current_position.copy()]

# Apply transformations step by step
for i, row in transformations.iterrows():
    translation_vector = np.array([row['translation_x'], row['translation_y'], row['translation_z']])
    rotation = row['rotation']

    # Apply rotation
    R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, rotation])
    rotated_translation_vector = np.dot(R, translation_vector.reshape(-1, 1)).flatten()

    # Print translation and rotation details
    print(f"Step {i + 1}:")
    print(f"Original translation vector: {translation_vector}")
    print(f"Rotated translation vector: {rotated_translation_vector}")
    print(f"Rotation (radians): {rotation}\n")

    # Apply translation
    current_position += rotated_translation_vector
    path_points.append(current_position.copy())

    # Create a line set for the current path
    current_path_points = np.array(path_points)
    lines = [[j, j + 1] for j in range(len(current_path_points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(current_path_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Create spheres (nodes) and coordinate frames at each path point
    spheres = []
    coordinate_frames = []
    for j, point in enumerate(current_path_points):
        # Create a sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(point)
        spheres.append(sphere)

        # Create a coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Rotate the coordinate frame 45 degrees around the Z-axis
        rotation_angle = np.pi / 4 * j  # 45 degrees in radians for each sphere
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation_angle])
        coord_frame.rotate(rotation_matrix, center=(0, 0, 0))  # Rotate around its own origin

        # Translate the coordinate frame to the sphere's position
        coord_frame.translate(point)

        coordinate_frames.append(coord_frame)

    # Combine all geometries for visualization
    geometries = [scan2, line_set] + spheres + coordinate_frames

    # Visualize the current step
    o3d.visualization.draw_geometries(geometries, window_name=f'Step {i + 1}')

print("Step-by-step visualization complete.")