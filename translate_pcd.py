import numpy as np
import open3d as o3d
import os
from functions_registration import *
import copy


def translate_pcd(input_name, output_name, start_side='Left'):
    # Define the folder path where your PLY files are stored and where the filtered PLY will be saved
    input_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\filtered_point_clouds'
    output_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\translated_point_clouds'

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Specify the PLY file name to load
    ply_file_path = os.path.join(input_folder, f"{input_name}.ply")

    # Step 1: Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # Make a deep copy of the point cloud for 2D conversion
    pcd_2d = copy.deepcopy(pcd)

    # Convert the points of the point cloud to a NumPy array
    point = np.asarray(pcd_2d.points)

    # Set the z-values to 0 (flatten onto xy-plane)
    point[:, 2] = 0

    # Convert the modified NumPy array back into a point cloud object
    pcd_2d.points = o3d.utility.Vector3dVector(point)

    # Print the modified points
    print("Points converted to 2D: ", np.asarray(pcd_2d.points))

    # Step 1: Convert the points of the point cloud to a NumPy array
    points = np.asarray(pcd_2d.points)

    if start_side == 'Left':
        # Step 2: Find the point with the lowest x value and lowest y value (separately)
        min_x_value = np.min(points[:, 0])
        min_y_value = np.min(points[:, 1])

        # Print the found values
        print(f"Lowest x value: {min_x_value}")
        print(f"Lowest y value: {min_y_value}")

        # Step 3: Create a new corner point at (min_x_value, min_y_value, 0)
        corner_point = np.array([min_x_value, min_y_value, 0])
        print(f"New corner point: {corner_point}")

        # Step 4: Translate the point cloud so that this new corner is at (0, 0)
        # Subtract the x and y values of the corner_point from all points
        points[:, 0] -= corner_point[0]  # Translate x values
        points[:, 1] -= corner_point[1]  # Translate y values

    else:
        # Step 2: Find the point with the lowest x value and lowest y value (separately)
        max_x_value = np.max(points[:, 0])
        min_y_value = np.min(points[:, 1])

        # Print the found values
        print(f"Lowest x value: {max_x_value}")
        print(f"Lowest y value: {min_y_value}")

        # Step 3: Create a new corner point at (min_x_value, min_y_value, 0)
        corner_point = np.array([max_x_value, min_y_value, 0])
        print(f"New corner point: {corner_point}")

        # Step 4: Translate the point cloud so that this new corner is at (0, 0)
        # Subtract the x and y values of the corner_point from all points
        points[:, 0] -= corner_point[0]  # Translate x values
        points[:, 1] -= corner_point[1]  # Translate y values

    # Step 5: Update the point cloud with the translated points
    pcd_2d.points = o3d.utility.Vector3dVector(points)

    # Step 6: Visualize the translated point cloud (optional)
    o3d.visualization.draw_geometries([pcd_2d])

    # Step 7: Save the translated point cloud to a new PLY file
    output_file = os.path.join(output_folder, f"{output_name}.ply")
    o3d.io.write_point_cloud(output_file, pcd_2d)

    print(f"Translation complete. The point cloud has been saved to: {output_file}")
    return pcd_2d

# translate_pcd('filtered_pcd1_08-10','translated_pcd1_08-10','Left')


