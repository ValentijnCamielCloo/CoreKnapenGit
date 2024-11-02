import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans, DBSCAN
from kneed import KneeLocator
from datetime import datetime
import pyvista as pv
import random
import csv
import constants as c
import pandas as pd
import logging
import time
import glob


# Configure logging
logging.basicConfig(
    filename='CORE_comparison.log',  # Log file name
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


def get_scan_date_time(scan_number):
    """
    Finds the scan file with the latest date in the 'scans' directory and retrieves
    the date and time of a specified scan file by its scan number.

    Parameters:
    - scan_number (int): The scan number to locate and retrieve the date and time for.

    Returns:
    - tuple: Formatted date (YYYY-MM-DD) and time (HH:MM:SS) of the specified scan.
             Returns (None, None) if no matching files are found.
    """
    # Initialize variables to track the latest date and target scan file
    latest_date = None  # Stores the latest date found across all scan files
    latest_date_file = None  # Filename with the latest date
    target_scan_file = None  # Filename matching the specified scan number

    # Loop through files in the scans folder to find the scan with the latest date
    for file in os.listdir('scans'):
        if file.endswith('_filtered.ply'):  # Check only for files ending with '_filtered.ply'
            parts = file.split('_')  # Split the filename by underscores

            try:
                # Extract the date part (format: ddmmyyyy) from the filename
                file_date_str = parts[2]
                file_date = datetime.strptime(file_date_str, "%d%m%Y").date()  # Convert to date object

                # Update if this file has a later date than the current latest date
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_date_file = file

                # Check if the file's scan number matches the desired scan number
                if parts[1] == str(scan_number):
                    target_scan_file = file  # Set the target file if the scan number matches

            except (IndexError, ValueError) as e:
                # Handle errors in parsing date or unexpected filename formats
                print(f"! Skipping file {file}, error parsing date: {e}")

    # Check if any files with a valid latest date were found
    if latest_date_file is None:
        print("! No scan files found with a valid date.")
        return None, None

    # Check if a file with the specified scan number was found
    if target_scan_file is None:
        print(f"! No scan file found with scan number {scan_number}.")
        return None, None

    # Extract date and time for the target scan file that matches the scan number
    try:
        target_parts = target_scan_file.split('_')  # Split target filename by underscores
        scan_date = target_parts[2]  # Extract the date (ddmmyyyy) component
        scan_time = target_parts[3]  # Extract the time (hhmmss) component

        # Convert extracted strings to date and time objects
        date_obj = datetime.strptime(scan_date, "%d%m%Y").date()
        time_obj = datetime.strptime(scan_time, "%H%M%S").time()

        # Format the date and time separately for CSV-friendly output
        formatted_date = date_obj.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
        formatted_time = time_obj.strftime("%H:%M:%S")  # Format as HH:MM:SS

        return formatted_date, formatted_time  # Return the formatted date and time

    except (IndexError, ValueError) as e:
        # Handle errors in parsing date or time from the target filename
        print("! Error parsing date and time from target file:", e)
        return None, None


def elbow_method(normals, save_path, max_k=10, show_figure=False):
    """
    Apply the elbow method to find the optimal number of clusters for the input data.

    Parameters:
    - normals (np.array): The dataset to be clustered (e.g., normal vectors of points).
    - save_path (str): The directory path where the elbow plot image will be saved.
    - max_k (int): The maximum number of clusters to consider (default is 10).

    Returns:
    - optimal_k (int): The optimal number of clusters determined by the elbow point.
    """
    # Initialize a list to store the inertia (sum of squared distances to the nearest cluster center)
    inertia_values = []

    # Define the range of cluster counts (k values) to test, from 1 up to max_k
    k_range = range(1, max_k + 1)

    # Loop over each k value and fit a KMeans model, storing the inertia for each k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(normals)
        inertia_values.append(kmeans.inertia_)  # Store the inertia (cluster compactness measure)

    # Use the KneeLocator to find the "elbow" point where inertia starts diminishing more slowly
    knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
    optimal_k = knee_locator.elbow

    if show_figure:
        # Plot the elbow graph to visualize inertia over different k values
        plt.plot(k_range, inertia_values, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method for Optimal k (Elbow at k={optimal_k})')

        # Highlight the elbow point on the plot with a vertical dashed red line
        plt.axvline(x=optimal_k, color='red', linestyle='--')

        # Save the elbow plot to the specified directory as a PNG file
        path = os.path.join(save_path, 'elbow_plot.png')
        plt.savefig(path, format='png', dpi=300)

        # Show the plot for immediate visualization
        plt.show()

    return optimal_k


def compute_rotation_matrix(source_vector, target_vector):
    """
    Find the rotation matrix that aligns source_vector to target_vector.

    Parameters:
    - source_vector (np.array): A 3D "source" vector to align.
    - target_vector (np.array): A 3D "target" vector that the source vector will be aligned to.

    Returns:
    - rotation_matrix (np.array): A 3x3 rotation matrix. When applied to source_vector, it aligns it with target_vector.
    """
    # Normalize the source and target vectors to unit vectors
    source_norm = (source_vector / np.linalg.norm(source_vector)).reshape(3)
    target_norm = (target_vector / np.linalg.norm(target_vector)).reshape(3)

    # Calculate the cross product of the normalized vectors, giving a perpendicular vector
    cross_product = np.cross(source_norm, target_norm)

    # Calculate the dot product of the normalized vectors to find the cosine of the angle between them
    dot_product = np.dot(source_norm, target_norm)

    # The sine of the angle between the vectors is the magnitude of the cross product vector
    sine_angle = np.linalg.norm(cross_product)

    # Construct the skew-symmetric cross-product matrix for vector v
    skew = np.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    # Calculate the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = (
            np.eye(3) +  # Identity matrix
            skew +  # Cross-product matrix (first-order rotation term)
            skew.dot(skew) * ((1 - dot_product) / (sine_angle ** 2))
    # Second-order rotation term to complete the rotation matrix
    )

    return rotation_matrix


class PointCloud:
    def __init__(self):
        """
        Initialize the PointCloud with the given point cloud file path.
        Automatically create an output directory with a unique number and timestamp
        inside the 'ProgressPilot' main directory.

        """
        # self.file_path = None
        self.save_counter = 1
        self.pcd = None

        # Create the main 'ProgressPilot' directory if it doesn't exist
        main_dir = "ProgressPilot"
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            logging.info(f"Main directory created: {main_dir}")

        # Get a list of all directories in the 'ProgressPilot' directory
        all_items = os.listdir(main_dir)

        # Filter to keep only directories
        existing_dirs = [item for item in all_items if os.path.isdir(os.path.join(main_dir, item))]

        # Further filter to keep only directories that match the pattern 'ProgressPilot_{number}_...'
        progress_dirs = []
        for directory in existing_dirs:
            if re.match(r"ProgressPilot_\d+_", directory):
                progress_dirs.append(directory)

        if progress_dirs:
            # Extract the number part from the directory names
            numbers = [int(re.search(r"ProgressPilot_(\d+)_", d).group(1)) for d in progress_dirs]
            next_number = max(numbers) + 1
        else:
            next_number = 1

        # Create the output directory name inside 'ProgressPilot'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(main_dir, f"ProgressPilot_{next_number}_{timestamp}")

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Output directory created: {self.output_dir}")

    def load_pcd(self, scan_dir):
        """
        Load the point cloud from the .ply file.
        """
        scan_dir = scan_dir
        self.pcd = []
        scan_files = sorted([f for f in os.listdir(scan_dir) if f.endswith('.ply')])
        # logging.info(f"Loading point cloud from {self.file_path}")
        for scan in scan_files:
            file_path = os.path.join(scan_dir, scan)
            pc = o3d.io.read_point_cloud(str(file_path))
            self.pcd.append(pc)

    def visualize(self, title=None, save_as_png=False, original_colors=True, rotate=False):
        """
        Visualize the current point clouds and optionally save the visualization as a PNG file.

        Parameters:
        - title (str): title of the visualization, and filename when save_as_png=True (default=None)
        - save_as_png (boolean): Optional to save as PNG file.
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            # if first_frame:
            #     self.pcd = [self.pcd[1]]

            plotter = pv.Plotter()

            for i, pc in enumerate(self.pcd):
                # Extract points and colors from Open3D point cloud and create PyVista point cloud
                points = np.asarray(pc.points)
                point_cloud = pv.PolyData(points)

                if pc.has_colors() and original_colors:
                    colors = np.asarray(pc.colors)  # Colors in Open3D are normalized (0-1)
                    colors = (colors * 255).astype(np.uint8)  # Convert to 0-255 for PyVista
                    if colors is not None:
                        point_cloud['RGB'] = colors  # Add color data to PyVista object
                    plotter.add_points(point_cloud, scalars='RGB', rgb=True)  # Plot with RGB colors

                else:
                    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']
                    plotter.add_points(point_cloud, color=colors[i])

            if title:
                plotter.add_title(title, font_size=12)

            plotter.zoom_camera(1.5)

            plotter.show(auto_close=False)



            if save_as_png:
                filename = title.replace(" ", "_") + ".png"
                save_path = os.path.join(self.output_dir, filename)
                plotter.screenshot(save_path)
                print(f"Visualization saved as {save_path}")

            if rotate:
                rotation_speed = 1
                display_time = 0.01
                # Rotate 360 degrees
                for _ in range(0, 360, rotation_speed):
                    plotter.camera.azimuth += rotation_speed  # Increment the azimuth angle
                    plotter.render()
                    time.sleep(display_time)

                # Add a one-second delay before closing
                time.sleep(1)

            # Close the window after rotation
            plotter.close()

        else:
            print("! No point cloud data to visualize.")

    def colorize(self):
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            # Get all .ply files in the directory
            ply_files = glob.glob(os.path.join('scans', "*.ply"))

            # Load the csv file to get the coordinates of the robot path
            csv_scan_scales = 'scancolor_scale.csv'
            file_path = os.path.join('scans', csv_scan_scales)
            scan_scales = pd.read_csv(file_path)
            scale_values = {row['Scan']: float(row['Scale']) for index, row in scan_scales.iterrows()}  # Ensure scale is float

            colorized = []
            for i, pc in enumerate(self.pcd):
                # Get the colors from the point cloud
                colors = np.asarray(pc.colors)

                # Get the scan name from the file name
                scan_name = "_".join(os.path.basename(ply_files[i]).split('_')[0:2])  # e.g., "Scan_4"

                # Get the corresponding scale value
                scale_value = scale_values.get(scan_name, 4.1)  # Default scale if not found is 4.1

                # Scale the colors
                colors_scaled = np.clip(colors * scale_value, 0, 1)
                pc.colors = o3d.utility.Vector3dVector(colors_scaled)

                # Filter out white colors
                non_white_mask = ~np.all(colors_scaled > 0.95, axis=1)
                non_white_colors = colors_scaled[non_white_mask]

                # Store red channel averages
                red_channel_values = []

                # Number of runs
                num_runs = 50
                for run in range(num_runs):
                    # K-means clustering
                    num_colors = 4
                    kmeans = KMeans(n_clusters=num_colors, random_state=run)
                    kmeans.fit(non_white_colors)

                    color_labels = kmeans.labels_
                    main_colors = kmeans.cluster_centers_

                    # Collect red channel averages
                    for i in range(num_colors):
                        avg_rgb = (main_colors[i] * 255).astype(int)
                        red_channel = avg_rgb[0]
                        red_channel_values.append(red_channel)

                # Estimate thresholds based on quantiles
                red_channel_values = np.array(red_channel_values)
                q1, q2, q3 = np.percentile(red_channel_values, [25, 50, 75])
                ranges = [
                    (0, q1),
                    (q1, q2),
                    (q2, q3),
                    (q3, 255)
                ]

                # Prepare lists to store red channel values for each assigned color
                color_hist_data = {
                    'Red': [],
                    'Dark Green': [],
                    'Yellow': [],
                    'Blue': [],
                }

                # Function to assign color based on red channel for scans [4, 6, 7, 8]
                def assign_color_based_on_red_standard(red_value):
                    if ranges[0][0] <= red_value < ranges[0][1]:
                        color_hist_data['Red'].append(red_value)
                        return (255, 0, 0)  # Assign red
                    elif ranges[1][0] <= red_value < ranges[1][1]:
                        color_hist_data['Dark Green'].append(red_value)
                        return (35, 157, 64)  # Assign dark green
                    elif ranges[2][0] <= red_value < ranges[2][1]:
                        color_hist_data['Yellow'].append(red_value)
                        return (255, 255, 68)  # Assign yellow
                    else:
                        color_hist_data['Blue'].append(red_value)
                        return (0, 0, 255)  # Assign blue

                # Function to assign color with yellow and green switched for scans [2, 3, 5, 9]
                def assign_color_based_on_red_switched(red_value):
                    if ranges[0][0] <= red_value < ranges[0][1]:
                        color_hist_data['Red'].append(red_value)
                        return (255, 0, 0)  # Assign red
                    elif ranges[1][0] <= red_value < ranges[1][1]:
                        color_hist_data['Yellow'].append(red_value)
                        return (255, 255, 68)  # Assign yellow (switched)
                    elif ranges[2][0] <= red_value < ranges[2][1]:
                        color_hist_data['Dark Green'].append(red_value)
                        return (35, 157, 64)  # Assign dark green (switched)
                    else:
                        color_hist_data['Blue'].append(red_value)
                        return (0, 0, 255)  # Assign blue

                # Apply the appropriate coloring function based on the scan number
                if int(scan_name.split("_")[1]) in [1, 2, 3, 5]:
                    assign_color_based_on_red = assign_color_based_on_red_switched
                else:
                    assign_color_based_on_red = assign_color_based_on_red_standard

                # Final coloring process based on average red channel values
                uniform_colors = np.zeros_like(colors)
                for run in range(num_runs):
                    # K-means clustering again for coloring
                    kmeans = KMeans(n_clusters=num_colors, random_state=run)
                    kmeans.fit(non_white_colors)

                    color_labels = kmeans.labels_
                    for i in range(num_colors):
                        avg_rgb = (kmeans.cluster_centers_[i] * 255).astype(int)
                        red_channel = avg_rgb[0]
                        cluster_points = (color_labels == i)
                        original_indices = np.where(non_white_mask)[0][cluster_points]
                        assigned_color = assign_color_based_on_red(red_channel)

                        uniform_colors[original_indices] = np.array(assigned_color) / 255.0  # Normalize for Open3D

                # Apply the colored array to the point cloud
                pc.colors = o3d.utility.Vector3dVector(uniform_colors)
                colorized.append(pc)

            self.pcd = colorized
            print(self.pcd)

            # Save
            # Define a path one level up from `output_folder`
            parent_folder = os.path.join(self.output_dir, '..', '..')

            # Resolve the relative path to an absolute path (optional but helps in complex scripts)
            parent_folder = os.path.abspath(parent_folder)

            # Create a subfolder within the parent directory (one level up) for saving files
            parent_subfolder = os.path.join(parent_folder, "colorized")
            os.makedirs(parent_subfolder, exist_ok=True)

            for i, pc in enumerate(self.pcd):
                # Now save files in this new directory
                file_path_in_parent = os.path.join(parent_subfolder, f"colorized_{i}.ply")
                # Save the point cloud as a .ply file using Open3D
                o3d.io.write_point_cloud(file_path_in_parent, pc)

                # self._save_ply('colorized')

        else:
            print('! No point clouds to colorize')

    def translate_orientate(self):
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            # Find the CSV file with format {yyyymmdd}_path_coordinates.csv
            csv_coordinates = None
            for file in os.listdir('robot'):
                if file.endswith('_path_coordinates.csv'):
                    csv_coordinates = file

            # Ensure a CSV file was found
            if csv_coordinates is None:
                print("! No path_coordinates file found.")
                return []

            # Load the csv file to get the coordinates of the robot path
            file_path = os.path.join('robot', csv_coordinates)
            path_coordinates = pd.read_csv(file_path)

            translated_orientated = []
            for i, pc in enumerate(self.pcd):
                # Get the coordinates and rotation for each scan from CSV
                x, y, z = path_coordinates.loc[i, ['x', 'y', 'z']]
                base_rotation = path_coordinates.loc[i, 'rotation']

                # Translate to the calculated coordinates
                pc.translate((x, y, z))

                # Apply an additional rotation for each scan based on the step
                additional_rotation_angle = np.pi / 4 * i  # 45 degrees per scan
                r_additional = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, additional_rotation_angle])
                pc.rotate(r_additional, center=(x, y, z))

                translated_orientated.append(pc)

            self.pcd = translated_orientated
            self._save_ply('translated_orientated')

        else:
            print('! No point clouds to translate and orientate')

    def estimate_normals(self, radius=0.1, max_nn=30, orientate_camera=False, orientate_not_middle=False, point_cloud=None, visualize_normals=False):
        """
        Estimate normals for all point clouds in self.pcd.
        Optionally flip the normals towards the camera / origin (if orientae_camera = True)

        Parameters:
        - radius (float): Radius for normal estimation.
        - max_nn (int): Maximum number of neighbors to consider for estimating normals.
        """
        if self.pcd:
            if point_cloud is None:
                point_cloud = self.pcd

            if type(point_cloud) is not list:
                point_cloud = [point_cloud]

            for pc in point_cloud:
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                if orientate_camera:
                    pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
                if orientate_not_middle:
                    # Find the CSV file with format {ddmmyyyy}_path_coordinates.csv
                    csv_coordinates = None
                    for file in os.listdir('robot'):
                        if file.endswith('_path_coordinates.csv'):
                            csv_coordinates = file

                    # Ensure a CSV file was found
                    if csv_coordinates is None:
                        print("! No path_coordinates file found.")
                        return []

                    # Load the CSV file with coordinates
                    file_path = os.path.join('robot', csv_coordinates)
                    path_coordinates = pd.read_csv(file_path)

                    x_middle = path_coordinates.iloc[0]['x']
                    y_middle = path_coordinates.iloc[2]['y']
                    z_middle = 0

                    centroid = np.array([x_middle, y_middle, z_middle])

                    print(f'centroid: {centroid}')
                    pc.orient_normals_towards_camera_location(camera_location=centroid)

                    # Flip the normals
                    normals_np = np.asarray(pc.normals)  # Convert to a NumPy array
                    normals_np = -normals_np  # Invert the normals
                    pc.normals = o3d.utility.Vector3dVector(normals_np)  # Convert back to Open3D Vector3dVector format


            if visualize_normals:
                o3d.visualization.draw_geometries(point_cloud, point_show_normal=True)


        else:
            print("! No point cloud data for calculating mean normal vector.")

    def voxel_downsample(self, voxel_size, point_cloud=None):
        """
        Downsample all point clouds using a voxel grid filter.

        Parameters:
        - voxel_size (float): The size of the voxel grid filter
        - point_cloud (PointCloud): use when not self.pcd is downsampled, but another point cloud
        """
        if self.pcd or point_cloud:
            if point_cloud is None:
                point_cloud = self.pcd

            if type(point_cloud) is not list:
                point_cloud = [point_cloud]

            print(f"\nDownsampling point clouds with voxel size {voxel_size}")
            pcd = []
            for i, pc in enumerate(point_cloud):
                pc = pc.voxel_down_sample(voxel_size=voxel_size)
                pcd.append(pc)

            self.pcd = pcd
            self._save_ply("downsampled")

        else:
            print("! No point cloud data to downsample.")

    def remove_outliers_radius(self, nb_points, radius):
        """
        Remove outliers from clusters using radius outlier removal.

        Parameters:
        - nb_points (int): minimum amount of points that the sphere should contain
        - radius (float): radius of the sphere that will be used for counting the neighbors
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            print('\nRemoving outliers within radius...')
            clean_pcd = []
            for pc in self.pcd:
                points_start = len(pc.points)
                cl, ind = pc.remove_radius_outlier(nb_points, radius)
                cleaned_pcd = pc.select_by_index(ind)
                clean_pcd.append(cleaned_pcd)

                points_end = len(cleaned_pcd.points)
                print(f'number of removed points: {points_start - points_end}')

            self.pcd = clean_pcd
            self._save_ply("radius_filtered")

        else:
            print("! No point cloud data to filter the outliers.")

    def remove_outliers_normal(self, radius, threshold_angle, max_nn):
        """
        Filter points whose normals deviate significantly from their local neighborhood normals.

        Parameters:
        - radius (float): Radius for local neighborhood search.
        - threshold_angle (float): Angular threshold (in degrees) for filtering points based on normals.
        - max_nn (int): Maximum number of neighbors to consider in the local search.
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            print('\nRemoving outliers based on normals...')
            filtered = []
            for pc in self.pcd:
                points_start = len(pc.points)
                # Ensure that the point cloud has normals computed
                if not pc.has_normals():
                    self.estimate_normals()

                # Create a KDTree for efficient neighbor search
                kdtree = o3d.geometry.KDTreeFlann(pc)

                points = np.asarray(pc.points)
                normals = np.asarray(pc.normals)

                # Store indices of points that meet the angular criteria
                indices_to_keep = []

                for i in range(len(points)):
                    # Search for neighbors within the specified radius
                    [k, idx, _] = kdtree.search_radius_vector_3d(pc.points[i], radius)

                    # Skip points with fewer than 3 neighbors
                    if k < 8:
                        continue

                    # Get the normals of the neighboring points
                    # Ensure idx is not empty before accessing it
                    neighbor_normals = normals[idx[:max_nn]] if len(idx) > 0 else []

                    if len(neighbor_normals) == 0:
                        continue  # Skip if no neighbors are found

                    # Compute the average normal of the neighbors
                    local_avg_normal = np.mean(neighbor_normals, axis=0)
                    local_avg_normal /= np.linalg.norm(local_avg_normal)  # Normalize the local average normal

                    # Compare the current point's normal with the local average normal
                    current_normal = normals[i]
                    current_normal /= np.linalg.norm(current_normal)  # Normalize the current normal

                    # Calculate the angle between the current normal and the local average normal
                    dot_product = np.dot(current_normal, local_avg_normal)
                    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors
                    angle = np.degrees(angle)

                    # Keep the point if the angle is within the threshold
                    if angle <= threshold_angle:
                        indices_to_keep.append(i)

                # Select the points that meet the normal consistency criterion
                filtered_pcd = pc.select_by_index(indices_to_keep)
                filtered.append(filtered_pcd)

                points_end = len(filtered_pcd.points)
                print(f'number of removed points: {points_start - points_end}')
                # o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)

            self.pcd = filtered
            self._save_ply('filtered_normals')

        else:
            print("! No point cloud data to filter the outliers.")

    def filter_colors(self, filter_color, color_threshold):
        """
        Filters out points in a point cloud based on a specified color.

        Parameters:
            filter_color (tuple): RGB color to filter out, as a tuple (r, g, b), where each value is between 0 and 1.
            color_threshold (float): Threshold for color filtering. Points within this range of `filter_color` will be removed.
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            filtered = []
            for pc in self.pcd:
                # Convert to numpy for easy color filtering
                points = np.asarray(pc.points)
                colors = np.asarray(pc.colors)

                # Create a mask for points that are NOT close to the specified filter color
                non_filtered_mask = ~((np.abs(colors[:, 0] - filter_color[0]) < color_threshold) &
                                      (np.abs(colors[:, 1] - filter_color[1]) < color_threshold) &
                                      (np.abs(colors[:, 2] - filter_color[2]) < color_threshold))

                # Apply the mask to filter points and colors
                filtered_points = points[non_filtered_mask]
                filtered_colors = colors[non_filtered_mask]

                # Create a new point cloud with filtered points and colors
                filtered_pc = o3d.geometry.PointCloud()
                filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
                filtered_pc.colors = o3d.utility.Vector3dVector(filtered_colors)

                filtered.append(filtered_pc)

            self.pcd = filtered
            self._save_ply('filtered_colors')

    def initial_alignment(self):
        """
        Rotate and translate the point clouds to align them in the coordinate system.

        """
        # Find the CSV file with format {ddmmyyyy}_path_coordinates.csv
        csv_coordinates = None
        for file in os.listdir('robot'):
            if file.endswith('_path_coordinates.csv'):
                csv_coordinates = file

        # Ensure a CSV file was found
        if csv_coordinates is None:
            print("! No path_coordinates file found.")
            return []

        # Load the csv file to get the coordinates of the robot path
        file_path = os.path.join('robot', csv_coordinates)
        path_coordinates = pd.read_csv(file_path)

        # Initialize scan list and other parameters
        scans_folder_path = 'scans'
        scan_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith('.ply')])

        # Initialize geometries list to visualize all scans in one go
        loaded_scan_files = []
        init_aligned_pcd = []
        coordinate_frames = []

        # Loop through each scan, set its position, and apply rotation
        for i, scan_file in enumerate(scan_files):
            self.load_pcd(scan_file)
            loaded_scan_files.append(self.pcd)

            # Get the coordinates and rotation for each scan from CSV
            x, y, z = path_coordinates.loc[i, ['x', 'y', 'z']]
            # base_rotation = path_coordinates.loc[i, 'rotation']

            # Translate to the calculated coordinates
            self.pcd.translate((x, y, z))

            # Apply an additional rotation for each scan based on the step
            additional_rotation_angle = np.pi / 4 * i  # 45 degrees per scan
            r_additional = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, additional_rotation_angle])
            self.pcd.rotate(r_additional, center=(x, y, z))

            # Visualize coordinate frame for each scan
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coord_frame.translate((x, y, z))
            coord_frame.rotate(r_additional, center=(x, y, z))  # Apply additional rotation to coordinate frame as well
            coordinate_frames.append(coord_frame)
            init_aligned_pcd.append(self.pcd)

        self._save_ply('scan', point_cloud=loaded_scan_files)
        # o3d.visualization.draw_geometries((init_aligned_pcd + coordinate_frames), window_name="Transformed Scans Visualization")
        self.pcd = init_aligned_pcd
        self._save_ply('transformed_scans')

    def registration(self):
        """
        Register different point clouds consecutively to create one 3D object.

        - sources_pcd (PointCloud) = a (list of) point cloud(s) which will be registered one by one
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            # Define multi-scale parameters for Colored ICP registration
            voxel_radius = [0.005, 0.005, 0.005]
            max_iter = [300, 80, 50]
            current_transformation = np.identity(4)

            # Initialize cumulative point cloud with the first scan
            cumulative_cloud = self.pcd[0]
            elevation_cloud = self.pcd[0]

            # Get all .ply files in the directory
            ply_files = glob.glob(os.path.join('scans', "*.ply"))
            cumulative_name = ply_files[0]

            # Export the initial cumulative cloud
            # initial_registered_path = os.path.join(output_folder, "Registered_0.ply")
            # o3d.io.write_point_cloud(initial_registered_path, cumulative_cloud)

            # Create CSV file to store registration details
            csv_file_path = os.path.join(self.output_dir, 'registered_path_coordinates.csv')

            # Open the CSV file outside the loop
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ["Scan_source", "Scan_target", "X", "Y", "Z", "Rotation_X", "Rotation_Y", "Rotation_Z"])

                # Load and register each scan iteratively
                for i in range(1, len(ply_files)):
                    # source_path = os.path.join(self.output_dir, ply_files[i])
                    source = self.pcd[i]
                    source_name = ply_files[i]
                    # print(f"Registering {cumulative_name} to {source_name}")

                    # Visualize the source cloud before registration
                    # initial_visualization = f"INITIAL ALIGNMENT: Source: {source_name}, Target: {cumulative_name}"
                    # o3d.visualization.draw_geometries([source, cumulative_cloud], window_name=initial_visualization)

                    # Multi-scale registration using Colored ICP
                    for scale in range(3):
                        iteration = max_iter[scale]
                        radius = voxel_radius[scale]
                        print(f"Scale {scale + 1} - Iterations: {iter}, Voxel size: {radius}")

                        # Before registration, show current counts
                        # print(
                            # f"Before registration: Cumulative cloud has {len(cumulative_cloud.points)} points, Source has {len(source.points)} points.")

                        # Downsample cumulative cloud and source cloud
                        cumulative_down = cumulative_cloud.voxel_down_sample(radius)
                        source_down = source.voxel_down_sample(radius)

                        # Estimate normals
                        cumulative_down.estimate_normals(
                            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))
                        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))

                        # Apply Colored ICP registration
                        result_icp = o3d.pipelines.registration.registration_colored_icp(
                            source_down, cumulative_down, radius, current_transformation,
                            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                              relative_rmse=1e-6,
                                                                              max_iteration=iteration))
                        current_transformation = result_icp.transformation
                        print(f"ICP Result for Scale {scale + 1}:")
                        print(f"  Transformation matrix:\n{current_transformation}")
                        print(f"  Fitness: {result_icp.fitness}, Inlier RMSE: {result_icp.inlier_rmse}")

                        # Visualize after each scale
                        scale_visualization = f"SCALE {scale + 1}: Source: {source_name}, Target: {cumulative_name} (after scale {scale + 1})"
                        transformed_source = source_down.transform(current_transformation)
                        # o3d.visualization.draw_geometries([transformed_source, cumulative_down],
                        #                                   window_name=scale_visualization)

                    # Transform the source to align with the cumulative point cloud
                    source.transform(current_transformation)

                    # Combine the cumulative cloud with the registered source
                    combined_cloud = cumulative_cloud + source

                    # Visualize the registered source and cumulative cloud
                    window_title_registration = f"REGISTERED: Source: {source_name}, Target: {cumulative_name} (after registration)"
                    # o3d.visualization.draw_geometries([combined_cloud], window_name=window_title_registration)

                    # Calculate the final translation and rotation
                    translation = current_transformation[:3, 3]
                    rotation = (
                        np.arctan2(current_transformation[1, 0], current_transformation[0, 0]),
                        np.arctan2(-current_transformation[2, 0],
                                   np.sqrt(current_transformation[2, 1] ** 2 + current_transformation[2, 2] ** 2)),
                        np.arctan2(current_transformation[2, 1], current_transformation[2, 2])
                    )

                    # Convert rotation from radians to degrees
                    rotation_degrees = tuple(np.degrees(rot) for rot in rotation)

                    # Print the translation and rotation
                    # print(
                    #     f"Translation of source '{source_name}' compared to target '{cumulative_name}': {translation}")
                    # print(
                    #     f"Rotation of source '{source_name}' compared to target '{cumulative_name}': {rotation_degrees}")

                    # Write the final translation and rotation to the CSV
                    csv_writer.writerow([source_name, cumulative_name, translation[0], translation[1], translation[2],
                                         *rotation_degrees])

                    # if i %2 == 0:     # Only add elevations, even scans
                    #     elevation_cloud = elevation_cloud + source
                    # Update cumulative_cloud and cumulative_name with the new registered source
                    cumulative_cloud = combined_cloud
                    # cumulative_name = f"Registered_{i}.ply"


            self.pcd = cumulative_cloud
            # Export the combined registered source
            self._save_ply('registered')
            # o3d.io.write_point_cloud(self.pcd)

        else:
            print("! No point cloud data for registration.")

    def cluster_kmeans_normals(self, max_k=10, remove_ground=False, biggest_cluster=False, show_elbow=False):
        """
        Cluster the point cloud based on normals using KMeans and the elbow method.

        Parameters:
        - mak_k (int) = maximum number of k to search for (default=10).
        - remove_ground (boolean) = filter out the cluster with normal pointing up (default=False).
        - biggest_cluster (boolean) = only keep the biggest cluster (default=False).
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            print('\nClustering point cloud based on normals...')
            clustering = []
            for pc in self.pcd:
                # Ensure that the point cloud has normals computed
                if not pc.has_normals():
                    self.estimate_normals(orientate_not_middle=True)

                normals = np.asarray(pc.normals)

                # Automatically choose the optimal number of clusters
                if show_elbow:
                    show_figure = True
                else:
                    show_figure = False
                n_clusters = elbow_method(normals, self.output_dir, max_k=max_k, show_figure=show_figure)
                print(f"Optimal k-clusters: {n_clusters}")

                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

                # Get cluster labels
                labels = kmeans.labels_
                max_label = labels.max()
                clusters = []
                cluster_length = []

                for i in range(max_label + 1):
                    indices = np.where(labels == i)[0]
                    cluster = pc.select_by_index(indices)
                    clusters.append(cluster)
                    cluster_length.append(len(cluster.points))

                if remove_ground:
                    non_upward_clusters = []
                    for i, cluster in enumerate(clusters):
                        # Calculate the mean normal of the cluster
                        mean_normal = np.mean(np.asarray(cluster.normals), axis=0)

                        if abs(mean_normal[2]) <= 0.5:  # Z-component threshold
                            non_upward_clusters.append(cluster)

                    clusters = non_upward_clusters  # Update clusters to only keep non-upward clusters

                if biggest_cluster:
                    max_length_index = np.argmax(cluster_length)
                    clusters = clusters[max_length_index]

                clustering.append(clusters)

            self.pcd = clustering
            if len(self.pcd) == 1:
                self.pcd = self.pcd[0]

            self._save_ply('cluster_kmeans')

        else:
            print("! No point cloud data for clustering.")

    def cluster_dbscan(self, eps, min_samples, biggest_cluster=True):
        """
        Cluster the point cloud using DBSCAN, which does not require specifying the number of clusters.

        Parameters:
        - eps (float): Maximum distance between two points to be considered in the same cluster.
        - min_samples (int): Minimum number of points in a neighborhood to form a core point.
        - biggest_cluster (bool): if True, only the biggest cluster is saved (default=True).
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]
            print(self.pcd)
            print('\nRemoving surrounding objects with DBSCAN clustering...')
            clustering = []
            for pc in self.pcd:
                points = np.asarray(pc.points)

                # Apply DBSCAN clustering
                dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

                # Get cluster labels
                labels = dbscan.labels_
                max_label = labels.max()
                clusters = []
                cluster_length = []

                for i in range(max_label + 1):
                    indices = np.where(labels == i)[0]
                    cluster = pc.select_by_index(indices)
                    clusters.append(cluster)
                    cluster_length.append(len(cluster.points))
                    # print(f"Cluster {i} has {len(cluster.points)} points.")

                if biggest_cluster:
                    max_length_index = np.argmax(cluster_length)
                    clusters = clusters[max_length_index]

                clustering.append(clusters)

                # print(f"Number of clusters after filtering: {len(clusters)}")

            self.pcd = clustering
            if len(self.pcd) == 1:
                self.pcd = self.pcd[0]

            self._save_ply('cluster_DBSCAN')

        else:
            print("! No point cloud data for clustering.")

    def orientate(self, meshes):
        """
        OVERBODIG GEWORDEN!!
        Orientate the point cloud in such a way that the orientation is the same as the model.
        This is the first step of aligning the point cloud and the model.

        Parameters:
        - meshes: model mesh, which is the target for orientation.
        """
        if self.pcd:
            print('\nRotate the point cloud to the same orientation as the model...')

            # Compute mean normal from one point cloud cluster
            if type(self.pcd) is list:
                pc = self.pcd[0]
            else:
                pc = self.pcd

            if not pc.has_normals():
                print('Computing normals...')
                self.estimate_normals(orientate_camera=True)

            normals = np.asarray(pc.normals)
            normal_source = np.mean(normals, axis=0)
            print(f'Normal source: {normal_source}')

            # Compute mean normal from one mesh plane
            if type(meshes) is list:
                mesh = meshes[0]
            else:
                mesh = meshes

            normals_mesh = mesh.point_data['Normals']
            normal_target = np.mean(normals_mesh, axis=0)
            print(f'Normal target: {normal_target}')

            rot_matrix = compute_rotation_matrix(normal_source, normal_target)
            print(f"Rotation Matrix:\n {rot_matrix}")

            # Perform rotation on the points and normals of all clusters
            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)

            rotated_points = points @ rot_matrix.T
            rotated_normals = normals @ rot_matrix.T

            # Update the point cloud with rotated points and normals
            pc.points = o3d.utility.Vector3dVector(rotated_points)
            pc.normals = o3d.utility.Vector3dVector(rotated_normals)

            self.pcd = pc
            self._save_ply('rotated')

        else:
            print("! No point cloud data for rotating.")

    def translate(self, dist_scanner_obj, height_scanner):
        # Find the CSV file with format {ddmmyyyy}_path_coordinates.csv
        csv_coordinates = None
        for file in os.listdir('robot'):
            if file.endswith('_path_coordinates.csv'):
                csv_coordinates = file

        # Ensure a CSV file was found
        if csv_coordinates is None:
            print("! No path_coordinates file found.")
            return []

        # Load the csv file to get the coordinates of the robot path
        file_path = os.path.join('robot', csv_coordinates)
        path_coordinates = pd.read_csv(file_path)

        # Get the coordinates and rotation for each scan from CSV
        x, y, z = path_coordinates.iloc[-2][['x', 'y', 'z']]
        x += dist_scanner_obj
        y += dist_scanner_obj
        z -= height_scanner
        path_corner_point = np.array([x, y, z])

        # Translate point clouds so the path corner point is at the origin
        translation_vector = -path_corner_point

        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            translated = []
            for pc in self.pcd:
                # Perform translation on the points and normals
                points = np.asarray(pc.points)
                normals = np.asarray(pc.normals)

                translated_points = points + translation_vector
                translated_normals = normals + translation_vector

                # Update the point cloud with rotated points and normals
                pc.points = o3d.utility.Vector3dVector(translated_points)
                pc.normals = o3d.utility.Vector3dVector(translated_normals)

                translated.append(pc)

            self.pcd = translated
            self._save_ply('translated')

        else:
            print("! No point cloud data for translation.")

    def _save_ply(self, file_name, point_cloud=None):
        """
        Save the point cloud(s) with the given file name to PLY files, in sequential order.

        Parameters:
        - file_name (str): The prefix for each PLY file name.
        - point_cloud (PointCloud): A PointCloud or list of PointCloud clusters to be saved. If None, the main point cloud is saved.
        """
        if point_cloud is None:
            point_cloud = self.pcd

        if point_cloud:
            if type(point_cloud) is not list:
                point_cloud = [point_cloud]

            if len(point_cloud) > 1:
                folder_name = f'{self.save_counter}_{file_name}'
                output_dir = os.path.join(self.output_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)

            else:
                output_dir = self.output_dir

            # Save each point cloud in the list with the current save counter prefix
            if len(point_cloud) > 1:
                for i, pc in enumerate(point_cloud):
                    save_path = os.path.join(output_dir, f"{file_name}_{i}.ply")
                    o3d.io.write_point_cloud(save_path, pc)
                    print(f"Saved: {save_path}")
            else:
                for pc in point_cloud:
                    save_path = os.path.join(output_dir, f"{self.save_counter}_{file_name}.ply")
                    o3d.io.write_point_cloud(save_path, pc)
                    print(f"Saved: {save_path}")

            # Increment the counter after each save
            self.save_counter += 1
        else:
            print("! No point cloud to save.")


class Mesh:
    def __init__(self):
        """
        Initialize the Mesh class.
        Automatically find the latest 'ProgressPilot' output directory to store results.

        """
        self.meshes = []

        # Find the latest created output directory inside 'ProgressPilot'
        main_dir = "ProgressPilot"
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Main directory created: {main_dir}")

        # Get all directories in 'ProgressPilot'
        all_items = os.listdir(main_dir)
        progress_dirs = [item for item in all_items if os.path.isdir(os.path.join(main_dir, item))]

        # Find the most recent directory based on timestamp
        if progress_dirs:
            progress_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(main_dir, d)), reverse=True)
            self.output_dir = os.path.join(main_dir, progress_dirs[0])
        else:
            raise Exception(f"No directories found in {main_dir}. Ensure that you first create a PointCloud project.")

    def load_meshes(self):
        """
        Load multiple meshes based on an Excel file that specifies which meshes to load, using the latest-dated Excel file.

        """
        # Find the latest Excel file with format {ddmmyyyy}_facade_order.csv
        latest_excel_file = None
        for file in os.listdir('model'):
            if file.endswith('_facade_order.csv'):
                if latest_excel_file is None or file > latest_excel_file:
                    latest_excel_file = file

        # Ensure an Excel file was found
        if latest_excel_file is None:
            print("! No facade order Excel file found.")
            return []

        # Extract the date from the file name
        date_str = latest_excel_file.split('_')[0]

        # Load the Excel file to get the order and facade columns
        file_path = os.path.join('model', latest_excel_file)
        facade_data = pd.read_csv(file_path)

        print(f"\nUsing the latest facade order file: {latest_excel_file}")

        # Loop through each facade specified in the Excel file
        for facade in facade_data['facade']:
            # Construct the mesh filename using the date and facade name
            mesh_file_name = f"{date_str}_{facade}.ply"
            mesh_file_path = os.path.join('model', mesh_file_name)

            # Check if the mesh file exists and load it
            if os.path.exists(mesh_file_path):
                mesh = pv.read(mesh_file_path)
                self.meshes.append(mesh)
                print(f"Loaded mesh: {mesh_file_name}")
            else:
                print(f"! Mesh file {mesh_file_name} does not exist in the directory.")

        self._save_meshes('model_mesh')

        return self.meshes

    def visualize(self, title=None, save_as_png=False, show_normals=False, rotate=False):
        """
        Visualize the current meshes and optionally save the visualization as a PNG file.

        Parameters:
        - title (str): title of the visualization, and filename when save_as_png=True (default=None)
        - save_as_png (boolean): Optional to save as PNG file.
        """
        if self.meshes:
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            plotter = pv.Plotter()

            for i, mesh in enumerate(self.meshes):
                if show_normals:
                    if mesh.n_points > 0 and 'Normals' not in mesh.point_data.keys():
                        mesh.compute_normals(inplace=True)

                    # Extract points and normals
                    points = mesh.points
                    mesh_normals = mesh.point_data['Normals']

                    # Create and add normals as cones
                    for point, normal in zip(points, mesh_normals):
                        # Create an arrow at each point along the normal
                        arrow = pv.Arrow(start=point, direction=normal, scale=0.02)
                        plotter.add_mesh(arrow, color='red')

                # Generate a random color for each mesh
                color = ['blue', 'orange', 'green', 'red', 'yellow']
                plotter.add_mesh(mesh, color=color[i], show_edges=True)

            if title:
                plotter.add_title(title, font_size=12)

            plotter.show(auto_close=False)

            if save_as_png:
                filename = title.replace(" ", "_") + ".png"
                save_path = os.path.join(self.output_dir, filename)
                plotter.screenshot(save_path)
                print(f"Mesh visualization saved as {save_path}")

            if rotate:
                rotation_speed = 1
                display_time = 0.01
                # Rotate 360 degrees
                for _ in range(0, 360, rotation_speed):
                    plotter.camera.azimuth += rotation_speed  # Increment the azimuth angle
                    plotter.render()
                    time.sleep(display_time)

                # Add a one-second delay before closing
                time.sleep(1)

            # Close the window after rotation
            plotter.close()

        else:
            print("! No meshes loaded to visualize.")

    def _save_meshes(self, file_name="mesh"):
        """
        Save all loaded meshes to the latest output directory with a given filename prefix.

        """
        if self.meshes:
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            if len(self.meshes) > 1:
                folder_name = file_name
                output_dir = os.path.join(self.output_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = self.output_dir

            for i, mesh in enumerate(self.meshes):
                save_path = os.path.join(output_dir, f"{file_name}_{i}.ply")
                mesh.save(save_path)
                print(f"Saved: {save_path}")

        else:
            print("! No meshes to save.")


class ComparePCDMesh:
    def __init__(self, point_cloud, meshes):
        """
        Initialize the Compare class with already loaded point clouds and meshes.

        - point_cloud: A PointCloud loaded and manipulated within the PointCloud class.
        - meshes: (A list of) Meshes loaded and manipulated within the Mesh class.
        """
        self.pcd = point_cloud
        self.meshes = meshes
        self.built = None
        self.not_built = None
        self.surface = None

        # Find the latest created output directory inside 'ProgressPilot'
        main_dir = "ProgressPilot"
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Main directory created: {main_dir}")

        # Get all directories in 'ProgressPilot'
        all_items = os.listdir(main_dir)
        progress_dirs = [item for item in all_items if os.path.isdir(os.path.join(main_dir, item))]

        # Find the most recent directory based on timestamp
        if progress_dirs:
            progress_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(main_dir, d)), reverse=True)
            self.output_dir = os.path.join(main_dir, progress_dirs[0])
        else:
            raise Exception(f"No directories found in {main_dir}. Ensure that you first create a PointCloud project.")

    def pair_pcd_mesh(self):
        """
        Every point cloud cluster belongs to a plane in the mesh.
        This method will find the pairs which belong together.

        """
        if self.pcd and self.meshes:
            pair = []
            for i, mesh in enumerate(self.meshes):
                # For every mesh find the corresponding pcd cluster, if there is one.
                normal_mesh = mesh.point_data['Normals']
                mean_normal_mesh = np.mean(normal_mesh, axis=0)

                # Normalize the mean normal of the mesh
                mean_normal_mesh /= np.linalg.norm(mean_normal_mesh)

                count_alignment = 0
                for j, pc in enumerate(self.pcd):
                    normals = np.asarray(pc.normals)
                    mean_normal_pc = np.mean(normals, axis=0)

                    # Normalize the mean normal of the point cloud
                    mean_normal_pc /= np.linalg.norm(mean_normal_pc)

                    alignment = np.dot(mean_normal_pc, mean_normal_mesh)
                    if alignment > 0.9:
                        print(f"Mesh {i} matches the cluster {j} with an alignment of {alignment:.2f}")
                        pair.append(pc)
                        count_alignment += 1
                    else:
                        continue

                if count_alignment == 0:
                    empty = None
                    pair.append(empty)

            self.pcd = pair

        else:
            print(f'! No meshes or point clouds are found for pairing')

    def check_bricks(self, points_per_brick):
        """
        Check which bricks are built based on point cloud data.

        Parameters:
        - points_per_brick (int): Minimum number of points required for a brick to be considered built.
        """
        if self.pcd and self.meshes:
            # Ensure self.pcd and self.meshes are lists
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            print('\nPairing the right point cloud to mesh...')
            self.pair_pcd_mesh()  # Pair point clouds to meshes

            print('\nChecking which bricks are built...')
            built = [None] * len(self.pcd)  # Initialize lists for built brick
            not_built = [None] * len(self.pcd)  # Initialize lists for not built bricks
            surface = [None] * len(self.pcd)  # Initialize surface mesh (bricks in model) list

            # Iterate over each surface mesh (brick in model) to check the amount of corresponding points
            for i, mesh in enumerate(self.meshes):
                normal_mesh = mesh.point_data['Normals']
                mean_normal = np.mean(normal_mesh, axis=0)

                # Extract surface mesh and number of components
                surface_mesh = mesh.extract_surface()
                surface[i] = surface_mesh
                n_surface_mesh = surface_mesh.n_cells
                print(f'Wall {i} - number of bricks in model: {n_surface_mesh}')

                if self.pcd[i] is not None:  # Ensure there's a point cloud to check
                    pc = self.pcd[i]
                    points = np.asarray(pc.points)

                    # Initialize lists for components with and without enough points
                    brick_enough_points = []
                    brick_not_enough_points = []

                    # Iterate over each surface mesh (brick in model)
                    for j in range(n_surface_mesh):
                        component = surface_mesh.extract_cells([j])

                        points_inside = 0  # Initialize count for points inside the component

                        # Check the mean normal to know in which coordinates (x, y or z) correspond with the bounds of the bricks
                        if (mean_normal[0] == 0) and (mean_normal[1] == 0):
                            x_bound, y_bound = component.bounds[:2], component.bounds[2:4]

                            # Loop through each point and check if it falls within the x and y bounds
                            for point in points:
                                px, py = point[0], point[1]

                                # Check if the point is within the x and y bounds
                                if x_bound[0] < px < x_bound[1] and y_bound[0] < py < y_bound[1]:
                                    points_inside += 1

                        elif (mean_normal[0] == 0) and (mean_normal[2] == 0):
                            x_bound, z_bound = component.bounds[:2], component.bounds[4:6]

                            # Loop through each point and check if it falls within the x and y bounds
                            for point in points:
                                px, pz = point[0], point[2]

                                # Check if the point is within the x and z bounds
                                if x_bound[0] < px < x_bound[1] and z_bound[0] < pz < z_bound[1]:
                                    points_inside += 1

                        elif (mean_normal[1] == 0) and (mean_normal[2] == 0):
                            y_bound, z_bound = component.bounds[2:4], component.bounds[4:6]

                            # Loop through each point and check if it falls within the x and y bounds
                            for point in points:
                                py, pz = point[1], point[2]

                                # Check if the point is within the y and z bounds
                                if y_bound[0] < py < y_bound[1] and z_bound[0] < pz < z_bound[1]:
                                    points_inside += 1
                        else:
                            print("! Something went wrong with finding the right bounds")

                        # Classify the bricks in built or not built based on the amount of points inside
                        if points_inside >= points_per_brick:
                            brick_enough_points.append(j)
                        else:
                            brick_not_enough_points.append(j)

                    built[i] = brick_enough_points
                    not_built[i] = brick_not_enough_points
                else:
                    continue  # Skip to the next mesh if no point cloud is available

            self.built = built  # Update built attribute
            self.not_built = not_built  # Update not built attribute
            self.surface = surface  # Update surface attribute

        else:
            print("! No meshes or point clouds were found for checking the bricks")

    def calculate_results(self):
        """
        Calculate and return the total number of built and not built bricks, along with progress percentage.

        - Return: Tuple containing total built bricks, total not built bricks, and progress percentage for each wall.
        """
        if self.built:
            print('\nCalculating the results...')
            n_bricks_total = []  # List for total built bricks per wall
            n_not_built_bricks_total = []  # List for total not built bricks per wall
            progress_total = []  # List for progress percentages per wall

            for i, bricks in enumerate(self.built):
                print(f'Wall {i}:')

                # Count the total number of built bricks
                if bricks:
                    n_bricks = len(bricks)
                else:
                    n_bricks = 0
                print(f'- Number of bricks built: {n_bricks}')
                n_bricks_total.append(n_bricks)

                # Count the total number of not built bricks
                if self.not_built[i]:
                    n_not_built_bricks = len(self.not_built[i])
                else:
                    surface = self.surface[i]
                    n_not_built_bricks = surface.n_cells
                print(f'- Number of bricks not built: {n_not_built_bricks}')
                n_not_built_bricks_total.append(n_not_built_bricks)

                # Calculate the progress of this specific wall if there are bricks
                if (n_bricks + n_not_built_bricks) > 0:
                    progress = round(n_bricks / (n_bricks + n_not_built_bricks) * 100, 2)
                else:
                    progress = 0
                print(f'- Progress: {progress} %')
                progress_total.append(progress)

            # Calculate the progress of all the walls together
            progress_sum = round(sum(n_bricks_total) / (sum(n_bricks_total) + sum(n_not_built_bricks_total)) * 100, 2)
            print(f'\nThe total progress: {progress_sum} %')

            return n_bricks_total, n_not_built_bricks_total, progress_total

        else:
            print('! There is no point cloud for calculating the results')

    def visualize(self, title=None, save_as_png=False, original_colors=True, rotate=False):
        """
        Visualize the current meshes and point clouds, and optionally save the visualization as a PNG file.

        Parameters:
        - title (str): title of the visualization, and filename when save_as_png=True (default=None).
        - save_as_png (boolean): Optional to save as PNG file (default=False).
        """
        if self.pcd or self.meshes:
            plotter = pv.Plotter()

            # Add meshes to the plot
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            for mesh in self.meshes:
                plotter.add_mesh(mesh, color='lightgrey', show_edges=True)

            # Add point clouds to the plot
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            for i, pc in enumerate(self.pcd):
                # Extract points and colors from Open3D point cloud and create PyVista point cloud
                points = np.asarray(pc.points)
                point_cloud = pv.PolyData(points)

                if pc.has_colors() and original_colors:
                    colors = np.asarray(pc.colors)  # Colors in Open3D are normalized (0-1)
                    colors = (colors * 255).astype(np.uint8)  # Convert to 0-255 for PyVista
                    if colors is not None:
                        point_cloud['RGB'] = colors  # Add color data to PyVista object
                    plotter.add_points(point_cloud, scalars='RGB', rgb=True)  # Plot with RGB colors

                else:
                    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']
                    plotter.add_points(point_cloud, color=colors[i])

            # Add a title if there is one
            if title:
                plotter.add_title(title, font_size=12)

            plotter.show(auto_close=False)

            # Save the image as png with title as filename if save_as_png=True
            if save_as_png:
                filename = title.replace(" ", "_") + ".png"  # Use the title as filename, but replace ' ' with '_'
                save_path = os.path.join(self.output_dir, filename)
                plotter.screenshot(save_path)
                print(f"Point cloud and mesh visualization saved as {save_path}")

            if rotate:
                rotation_speed = 1
                display_time = 0.01
                # Rotate 360 degrees
                for _ in range(0, 360, rotation_speed):
                    plotter.camera.azimuth += rotation_speed  # Increment the azimuth angle
                    plotter.render()
                    time.sleep(display_time)

                # Add a one-second delay before closing
                time.sleep(1)

            # Close the window after rotation
            plotter.close()

        else:
            print("! No point clouds or meshes loaded to visualize.")

    def visualize_result(self, filename_vis, title='Results', save_as_png=True, rotate=False):
        """
        Visualize the results and optionally save the visualization as a PNG file.
        Add information to the image on the progress of the total wall.

        Parameters:
        - title (str): title of the visualization, and filename when save_as_png=True (default='Results').
        - save_as_png (boolean): Optional to save as PNG file (default=True).
        """
        if self.pcd or self.meshes:
            # Calculate the results, extract the cells which are built
            n_bricks_total, n_not_built_bricks_total, progress_total = self.calculate_results()

            plotter = pv.Plotter()

            # Add the surface meshes (bricks) to the visualization when it is built
            for i, surface in enumerate(self.surface):
                built_indices = self.built[i]
                if built_indices is None:  # If there are no built bricks in a wall, continue with the next wall
                    continue

                built_bricks = surface.extract_cells(
                    built_indices)  # Extract the right surface meshes (bricks) based on indices
                plotter.add_mesh(built_bricks, color='green', opacity=1, show_edges=True)

            # Load the entire wall for visualisation, this shows the wall in 3D
            file_path_vis = os.path.join("model", filename_vis)
            mesh_vis = pv.read(file_path_vis)
            plotter.add_mesh(mesh_vis, color='red', opacity=0.2)

            # Calculate the progress of the entire wall
            progress = round(sum(n_bricks_total) / (sum(n_bricks_total) + sum(n_not_built_bricks_total)) * 100, 2)

            # Create a single multiline string for the text to add to the visualization
            multi_line_text = (
                f"Progress = {progress} %\n"
                f"- built = {sum(n_bricks_total)} bricks\n"
                f"- to be built = {sum(n_not_built_bricks_total)} bricks"  # Adjust this based on your logic
            )
            plotter.add_text(multi_line_text, font_size=10)

            # Add a title if there is one
            if title:
                plotter.add_title(title, font_size=12)

            plotter.show(auto_close=False)

            # Save the image as png with title as filename if save_as_png=True
            if save_as_png:
                filename = title.replace(" ", "_") + ".png"  # Use the title as filename, but replace ' ' with '_'
                save_path = os.path.join(self.output_dir, filename)
                plotter.screenshot(save_path)
                print(f"Results saved as {save_path}")

            if rotate:
                rotation_speed = 1
                display_time = 0.03
                # Rotate 360 degrees
                for _ in range(0, 360, rotation_speed):
                    plotter.camera.azimuth += rotation_speed  # Increment the azimuth angle
                    plotter.render()
                    time.sleep(display_time)

                # Add a one-second delay before closing
                time.sleep(1)

            # Close the window after rotation
            plotter.close()

        else:
            print("! No point clouds or meshes loaded to visualize.")

    def write_results(self):
        """
        Export the calculated results to a CSV file.

        """
        # Calculate the results
        n_bricks, n_no_bricks, progress = self.calculate_results()

        # Check the data type, if it is not a list, change it to list
        if type(n_bricks) is not list:
            n_bricks = [n_bricks]
        if type(n_no_bricks) is not list:
            n_no_bricks = [n_no_bricks]
        if type(progress) is not list:
            progress = [progress]

        filename = 'Results.csv'
        save_path = os.path.join(self.output_dir, filename)

        # Find corresponding date and time of each scan
        date = []
        time = []
        for file in os.listdir('scans'):
            if file.endswith('_filtered.ply'):  # Check only for files ending with '_filtered.ply'
                parts = file.split('_')

                # Extract and format the date part (format: yyyymmdd)
                scan_date_str = parts[2]
                scan_date = datetime.strptime(scan_date_str, "%Y%m%d").date()
                formatted_date = scan_date.strftime("%Y-%m-%d")
                date.append(formatted_date)

                # Extract and format the time part (format: hhmmss)
                scan_time_str = parts[3]
                scan_time = datetime.strptime(scan_time_str, "%H%M%S").time()
                formatted_time = scan_time.strftime("%H:%M:%S")
                time.append(formatted_time)

        # Find corresponding facade to each wall number in the facade order file
        latest_excel_file = None
        for file in os.listdir('model'):
            if file.endswith('_facade_order.csv'):
                if latest_excel_file is None or file > latest_excel_file:
                    latest_excel_file = file

        # Ensure an Excel file was found
        if latest_excel_file is None:
            print("! No facade order Excel file found.")
            return []

        # Load the Excel file to get the facade column and add it to a list
        file_path = os.path.join('model', latest_excel_file)
        facade_data = pd.read_csv(file_path, sep=';')
        facades = facade_data['facade'].tolist()

        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['wall', 'scan date', 'scan time',
                             'facade', 'bricks built',
                             'bricks to built', 'progress (%)'])

            # Iterate over the results and write each row
            for i in range(len(n_bricks)):
                writer.writerow([i, date[i], time[i], facades[i], n_bricks[i], n_no_bricks[i], progress[i]])

            # Calculate the total progress
            progress_total = round(sum(n_bricks) / (sum(n_bricks) + sum(n_no_bricks)) * 100, 2)
            writer.writerow(['total', '', '', '', sum(n_bricks), sum(n_no_bricks), progress_total])

        print(f"\nResults written to {save_path}")
