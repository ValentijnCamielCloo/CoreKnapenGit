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


def elbow_method(normals, save_path, max_k=10):
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
            skew.dot(skew) * ((1 - dot_product) / (sine_angle ** 2))  # Second-order rotation term to complete the rotation matrix
    )

    return rotation_matrix


class PointCloud:
    def __init__(self, file_name_pcd):
        """
        Initialize the PointCloud with the given point cloud file path.
        Automatically create an output directory with a unique number and timestamp
        inside the 'ProgressPilot' main directory.
        """
        self.file_name_pcd = file_name_pcd
        self.save_counter = 1
        self.pcd = None

        # Define file path as relative to the 'scans' folder
        self.file_path = os.path.join("scans", file_name_pcd)

        # Create the main 'ProgressPilot' directory if it doesn't exist
        main_dir = "ProgressPilot"
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Main directory created: {main_dir}")

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
            print(f"Output directory created: {self.output_dir}")

    def load_pcd(self):
        """
        Load the point cloud from the .ply file.
        """
        print(f"\nLoading point cloud from {self.file_path}")
        self.pcd = o3d.io.read_point_cloud(str(self.file_path))
        self._save_ply("scan")

    def visualize(self, save_as_png=False, filename='visualization'):
        """
        Visualize the current point clouds and optionally save the visualization as a PNG file.

        Parameters:
        - save_as_png (boolean): Optional to save as PNG file.
        - filename (str): When saved as PNG, this will be the file name.
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            vis = o3d.visualization.Visualizer()
            vis.create_window()

            # # Create a coordinate frame
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # vis.add_geometry(axis)

            if len(self.pcd) > 1:
                for pc in self.pcd:
                    random_color = [random.random(), random.random(), random.random()]
                    pc.paint_uniform_color(random_color)  # Paint point cloud with random color
                    vis.add_geometry(pc)
            else:
                for pc in self.pcd:
                    vis.add_geometry(pc)

            # Set a custom camera position
            ctr = vis.get_view_control()

            # Example settings for front view
            ctr.set_lookat([0, 0, 0])  # Set the focal point (center of the point cloud)
            ctr.set_front([0, -1, 0.5])  # Set the camera direction to view from the front
            ctr.set_up([0, 0, 1])  # Set the "up" direction
            ctr.set_zoom(0.4)  # Adjust the zoom level if needed

            vis.run()  # Run the visualizer

            if save_as_png:
                filename = f'{filename}.png'
                save_path = os.path.join(self.output_dir, filename)
                vis.capture_screen_image(save_path)
                print(f"\nVisualization saved as {save_path}")

            vis.destroy_window()

        else:
            print("! No point cloud data to visualize.")

    def estimate_normals(self, radius=0.1, max_nn=30, orientate_camera=False):
        """
        Estimate normals for all point clouds in self.pcd.
        Optionally flip the normals towards the camera / origin (if orientae_camera = True)

        Parameters:
        - radius (float): Radius for normal estimation.
        - max_nn (int): Maximum number of neighbors to consider for estimating normals.
        """
        if self.pcd:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            for pc in self.pcd:
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                if orientate_camera:
                    pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
            # o3d.visualization.draw_geometries(self.pcd, point_show_normal=True)

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

        else:
            print("! No point cloud data to filter the outliers.")

    def registration(self, source_pcd):
        """
        Register different point clouds consecutively to create one 3D object.

        - sources_pcd (PointCloud) = a (list of) point cloud(s) which will be registered one by one
        """
        if self.pcd:
            target = self.pcd
            if type(source_pcd) is not list:
                source_pcd = [source_pcd]
            for source in source_pcd:
                self.voxel_downsample(voxel_size=c.VOXEL_SIZE, pcd=source)
                # global_registration(target, source)   # Call function outside class
                # local_registration(target, source)   # Call function outside class
                target = target + source
            self.pcd = target

        else:
            print("! No point cloud data for registration.")

    def cluster_kmeans_normals(self, max_k=10, remove_ground=True, biggest_cluster=False):
        """
        Cluster the point cloud based on normals using KMeans and the elbow method.
        """
        if self.pcd:
            if type(self.pcd) is list:
                merged_pcd = self.pcd[0]
                for pc in self.pcd[1:]:
                    merged_pcd += pc
                self.pcd = merged_pcd

            print('\nClustering point cloud based on normals...')
            # Ensure that the point cloud has normals computed
            if not self.pcd.has_normals():
                self.estimate_normals()

            normals = np.asarray(self.pcd.normals)

            # Automatically choose the optimal number of clusters
            n_clusters = elbow_method(normals, self.output_dir, max_k=max_k)
            print(f"Optimal k-clusters: {n_clusters}")

            # Apply KMeans clustering
            clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

            # Get cluster labels
            labels = clustering.labels_
            max_label = labels.max()
            clusters = []
            cluster_length = []

            for i in range(max_label + 1):
                indices = np.where(labels == i)[0]
                cluster = self.pcd.select_by_index(indices)
                clusters.append(cluster)
                cluster_length.append(len(cluster.points))

            if remove_ground:
                non_upward_clusters = []
                for i, cluster in enumerate(clusters):
                    # Calculate the mean normal of the cluster
                    mean_normal = np.mean(np.asarray(cluster.normals), axis=0)

                    if mean_normal[2] <= 0.5:  # Z-component threshold
                        non_upward_clusters.append(cluster)

                clusters = non_upward_clusters  # Update clusters to only keep non-upward clusters

            if biggest_cluster:
                max_length_index = np.argmax(cluster_length)
                clusters = clusters[max_length_index]
                clusters = [clusters]

            print(f"Number of clusters after filtering: {len(clusters)}")

            self.pcd = clusters
            self._save_ply('cluster_kmeans')

        else:
            print("! No point cloud data for clustering.")

    def cluster_dbscan(self, eps, min_samples, biggest_cluster=True):
        """
        Cluster the point cloud using DBSCAN, which does not require specifying the number of clusters.

        Parameters:
        - eps (float): Maximum distance between two points to be considered in the same cluster.
        - min_samples (int): Minimum number of points in a neighborhood to form a core point.
        - remove_small_clusters (bool): Whether to remove small clusters.
        """
        if self.pcd:
            if type(self.pcd) is list:
                merged_pcd = self.pcd[0]
                for pc in self.pcd[1:]:
                    merged_pcd += pc
                self.pcd = merged_pcd

            print('\nRemoving surrounding objects with DBSCAN clustering...')
            points = np.asarray(self.pcd.points)

            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

            # Get cluster labels
            labels = clustering.labels_
            max_label = labels.max()
            clusters = []
            cluster_length = []

            for i in range(max_label + 1):
                indices = np.where(labels == i)[0]
                cluster = self.pcd.select_by_index(indices)
                clusters.append(cluster)
                cluster_length.append(len(cluster.points))
                # print(f"Cluster {i} has {len(cluster.points)} points.")

            if biggest_cluster:
                max_length_index = np.argmax(cluster_length)
                clusters = clusters[max_length_index]
                clusters = [clusters]

            print(f"Number of clusters after filtering: {len(clusters)}")

            self.pcd = clusters
            self._save_ply('cluster_DBSCAN')

        else:
            print("! No point cloud data for clustering.")

    def orientate(self, meshes):
        """
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

    def translate(self):
        """
        Translate the point cloud to the origin, in which the model is also located.
        This is the second step of aligning the point cloud and the model.

        """
        if self.pcd:
            if type(self.pcd) is list:
                merged_pcd = self.pcd[0]
                for pc in self.pcd[1:]:
                    merged_pcd += pc
                self.pcd = merged_pcd
            pc = self.pcd

            print('\nTranslate the point cloud towards the origin (0,0)...')
            # Find the minimal coordinates to create a new corner
            min_point = pc.get_min_bound()
            min_x = min_point[0]
            min_y = min_point[1]
            min_z = min_point[2]

            corner_point = np.array([min_x, min_y, min_z])
            print(f'New corner point: {corner_point}')

            # Translate both point clouds so the new corner point is at the origin
            translation_vector = -corner_point

            # Perform translation on the points and normals
            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)

            translated_points = points + translation_vector
            translated_normals = normals + translation_vector

            # Update the point cloud with rotated points and normals
            pc.points = o3d.utility.Vector3dVector(translated_points)
            pc.normals = o3d.utility.Vector3dVector(translated_normals)

            self.pcd = pc
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

            # Save each point cloud in the list with the current save counter prefix
            if len(point_cloud) > 1:
                for i, pc in enumerate(point_cloud):
                    save_path = os.path.join(self.output_dir, f"{self.save_counter}_{file_name}_{i}.ply")
                    o3d.io.write_point_cloud(save_path, pc)
                    print(f"Saved: {save_path}")
            else:
                for pc in point_cloud:
                    save_path = os.path.join(self.output_dir, f"{self.save_counter}_{file_name}.ply")
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
        facade_data = pd.read_csv(file_path, sep=';')

        print(f"\nUsing the latest facade order file: {latest_excel_file}")

        # Prepare list to store the loaded meshes
        self.meshes = []

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

    def visualize(self, save_as_png=False, filename='mesh_visualization'):
        """
        Visualize the loaded meshes and optionally save the visualization as a PNG file.

        Parameters:
        - save_as_png (boolean): Optional to save as PNG file.
        - filename (str): When saved as PNG, this will be the file name.
        """
        if self.meshes:
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            plotter = pv.Plotter()

            if len(self.meshes) > 1:
                for mesh in self.meshes:
                    # Generate a random color for each mesh
                    color = np.random.rand(3)  # Random color [R, G, B]
                    plotter.add_mesh(mesh, color=color, show_edges=True)
            else:
                for mesh in self.meshes:
                    plotter.add_mesh(mesh)

            plotter.show()

            if save_as_png:
                filename = f'{filename}.png'
                plotter.screenshot(filename)
                print(f"Mesh visualization saved as {filename}")

        else:
            print("! No meshes loaded to visualize.")

    def visualize_normals(self):
        """
        Load multiple PLY files, estimate normals if not present, and visualize each mesh and its normals.

        """
        if self.meshes:
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            # Create a PyVista Plotter
            plotter = pv.Plotter()

            for mesh in self.meshes:
                # Check if the mesh has normals
                if mesh.n_points > 0 and 'Normals' not in mesh.point_data.keys():
                    mesh.compute_normals(inplace=True)

                # Extract points and normals
                points = mesh.points
                normals = mesh.point_data['Normals']

                # Add the mesh to the plotter
                plotter.add_mesh(mesh, color='lightblue', show_edges=True)

                # Create and add normals as cones
                for point, normal in zip(points, normals):
                    # Create a cone at each point along the normal
                    arrow = pv.Arrow(start=point, direction=normal, scale=0.01)
                    plotter.add_mesh(arrow, color='red')

            # Step 7: Show the plot
            plotter.show()
        else:
            print("! No meshes loaded to visualize the normals.")

    def _save_meshes(self, file_name="mesh"):
        """
        Save all loaded meshes to the latest output directory with a given filename prefix.
        """
        if self.meshes:
            for i, mesh in enumerate(self.meshes):
                save_path = os.path.join(self.output_dir, f"{file_name}_{i}.ply")
                mesh.save(save_path)
                # print(f"Saved mesh {i} as {save_path}")
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

    def pair_PCD_mesh(self):
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
            print(f'self.pcd = {self.pcd}')
            print(f'shape self.pcd = {np.shape(self.pcd)}')

        else:
            print(f'! No meshes or point clouds are found for pairing')


    def check_bricks(self, points_per_brick):
        if self.pcd and self.meshes:
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            print('\nPairing the right point cloud to mesh...')
            self.pair_PCD_mesh()

            print('\nChecking which bricks are built...')
            built = [None]*len(self.pcd)
            print(f'built: {built}')
            not_built = [None]*len(self.pcd)
            surface = [None]*len(self.pcd)

            for i, mesh in enumerate(self.meshes):
                normal_mesh = mesh.point_data['Normals']
                mean_normal = np.mean(normal_mesh, axis=0)

                # Extract surface mesh and number of components
                surface_mesh = mesh.extract_surface()
                surface[i] = surface_mesh
                n_components = surface_mesh.n_cells
                print(f'Wall {i} - number of bricks in model: {n_components}')

                if self.pcd[i] is not None:
                    pc = self.pcd[i]
                    points = np.asarray(pc.points)

                    # Initialize lists for components with and without enough points
                    brick_enough_points = []
                    brick_not_enough_points = []

                    # Iterate over each brick in the mesh
                    for j in range(n_components):
                        component = surface_mesh.extract_cells([j])

                        points_inside = 0

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
                    # built.append(brick_enough_points)
                    # not_built.append(brick_not_enough_points)
                else:
                    continue

            self.built = built
            # print(f'self.built = {self.built}')
            self.not_built = not_built
            # print(f'self.not_built = {self.not_built}')
            self.surface = surface
            # print(f'self.surface = {self.surface}')

        else:
            print("! No meshes or point clouds were found for checking the bricks")

    def visualize(self, save_as_png=False, filename='compare_visualization'):
        if self.pcd:
            plotter = pv.Plotter()

            # Add meshes
            if type(self.meshes) is not list:
                self.meshes = [self.meshes]

            for mesh in self.meshes:
                plotter.add_mesh(mesh, color='lightgrey', show_edges=True)

            # Add point clouds
            if type(self.pcd) is not list:
                self.pcd = [self.pcd]

            for pc in self.pcd:
                points_pc = np.asarray(pc.points)
                pc = pv.PolyData(points_pc)
                plotter.add_mesh(pc, color='lightblue')

            plotter.show()

            if save_as_png:
                filename = f'{filename}.png'
                plotter.screenshot(filename)
                print(f"Point cloud and mesh visualization saved as {filename}")

        else:
            print("! No point clouds loaded to visualize.")

    def calculate_results(self):

        if self.built:
            print('\nCalculating the results...')
            n_bricks_total = []
            n_not_built_bricks_total = []
            progress_total = []

            for i, bricks in enumerate(self.built):
                print(f'Wall {i}:')
                if bricks:
                    n_bricks = len(bricks)
                else:
                    n_bricks = 0
                print(f'- Number of bricks built: {n_bricks}')
                n_bricks_total.append(n_bricks)

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


            progress = round(sum(n_bricks_total) / (sum(n_bricks_total) + sum(n_not_built_bricks_total)) * 100, 2)
            print(f'\nThe total progress: {progress} %')

            return n_bricks_total, n_not_built_bricks_total, progress_total

        else:
            print('! There is no point cloud for calculating the results')

    def write_results(self):
        n_bricks, n_no_bricks, progress = self.calculate_results()
        if type(n_bricks) is not list:
            n_bricks = [n_bricks]
        if type(n_no_bricks) is not list:
            n_no_bricks = [n_no_bricks]
        if type(progress) is not list:
            progress = [progress]
        # Open the CSV file in write mode ('w')
        file_name = 'results.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['bricks built', 'bricks to be built', 'progress (%)'])

            # Iterate over both lists simultaneously and write rows to the CSV
            for n_bricks, n_no_bricks, progress in zip(n_bricks, n_no_bricks, progress):
                writer.writerow([n_bricks, n_no_bricks, progress])

        print(f"Data written to {file_name}")

    def visualize_result(self, file_name_vis):
        # Calculate the results, extract the cells which are built
        n_bricks_total, n_not_built_bricks_total, progress_total = self.calculate_results()

        # Visualize the mesh and components
        plotter = pv.Plotter()

        for i, surface in enumerate(self.surface):
            built_indices = self.built[i]
            if built_indices is None:
                continue

            built_bricks = surface.extract_cells(built_indices)
            plotter.add_mesh(built_bricks, color='green', opacity=1, show_edges=True)

        # Load the mesh for visualisation
        file_path_vis = os.path.join("model", file_name_vis)
        mesh_vis = pv.read(file_path_vis)
        plotter.add_mesh(mesh_vis, color='red', opacity=0.8, show_edges=True)

        progress = round(sum(n_bricks_total) / (sum(n_bricks_total) + sum(n_not_built_bricks_total)) * 100, 2)
        plotter.add_text(f"Progress = {progress} %")

        # plotter.camera_position = 'xz'
        plotter.camera.position = (-1, -1, 0.5)
        plotter.camera.focal_point = (0, 0, 0.1)
        plotter.camera.viewup = (0, 0, 1)
        plotter.camera.zoom(1.7)

        plotter.show()