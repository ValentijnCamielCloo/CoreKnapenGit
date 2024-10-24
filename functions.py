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

def elbow_method(normals, save_path, max_k=10):
    """
    Apply the elbow method to find the optimal number of clusters.
    """
    print("Running elbow method to determine optimal number of clusters...")
    inertia_values = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(normals)
        inertia_values.append(kmeans.inertia_)

    # Find the elbow point
    knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
    optimal_k = knee_locator.elbow

    # Plot the elbow graph
    plt.plot(k_range, inertia_values, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for Optimal k (Elbow at k={optimal_k})')
    plt.axvline(x=optimal_k, color='red', linestyle='--')
    path = os.path.join(save_path, 'elbow_plot.png')
    plt.savefig(path, format='png', dpi=300)
    plt.show()

    return optimal_k

def compute_rotation_matrix(source_vector, target_vector):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (source_vector / np.linalg.norm(source_vector)).reshape(3), (target_vector / np.linalg.norm(target_vector)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix




class PointCloud:
    def __init__(self, file_dir_pcd, file_name_pcd):
        """
        Initialize the PointCloud with the given point cloud file path.
        Automatically create an output directory with a unique number and timestamp
        inside the 'ProgressPilot' main directory.
        """
        self.file_dir_pcd = file_dir_pcd
        self.file_name_pcd = file_name_pcd
        self.pcd = None

        self.file_path = os.path.join(file_dir_pcd, file_name_pcd)

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
        print(f"Loading point cloud from {self.file_path}")
        self.pcd = o3d.io.read_point_cloud(str(self.file_path))
        self._save_ply("scan")
        return self.pcd

    def visualize(self, save_as_png=False, filename='visualization'):
        """
        Visualize the current point clouds and optionally save the visualization as a PNG file.
        """

        if self.pcd:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            if type(self.pcd) is list:
                for pc in self.pcd:
                    random_color = [random.random(), random.random(), random.random()]
                    pc.paint_uniform_color(random_color)  # Paint point cloud with random color
                    vis.add_geometry(pc)
            else:
                vis.add_geometry(self.pcd)

            vis.run()  # Run the visualizer

            if save_as_png:
                filename = f'{filename}.png'
                save_path = os.path.join(self.output_dir, filename)
                vis.capture_screen_image(save_path)
                print(f"Visualization saved as {save_path}")

            vis.destroy_window()

        else:
            print("No point cloud data to visualize.")

    def estimate_normals(self, radius=0.1, max_nn=30):
        """
        Estimate normals for all point clouds in self.pcd.

        :param radius: Radius for normal estimation.
        :param max_nn: Maximum number of neighbors to consider for estimating normals.
        """
        if self.pcd:
            if type(self.pcd) is list:
                mean_normals = []
                for pc in self.pcd:
                    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            else:
                self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        else:
            print("No point cloud data for calculating mean normal vector.")


    def voxel_downsample(self, voxel_size):
        """
        Downsample all point clouds using a voxel grid filter.

        Parameters:
        voxel_size (float): The size of the voxel grid filter
        """

        if self.pcd:
            print(f"Downsampling point clouds with voxel size {voxel_size}")

            if type(self.pcd) is list:
                point_cloud = []
                for i, pc in enumerate(self.pcd):
                    self.pcd = pc.voxel_down_sample(voxel_size=voxel_size)
                    point_cloud.append(self.pcd)
                self.pcd = point_cloud

            else:
                self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)

            self._save_ply("downsampled")

        else:
            print("No point cloud data to downsample.")

    def remove_outliers_radius(self, nb_points, radius):
        """
        Remove outliers from clusters using radius outlier removal.

        Parameters:
        nb_points (int): minimum amount of points that the sphere should contain
        radius (float): radius of the sphere that will be used for counting the neighbors
        """
        if self.pcd:
            print('Removing outliers...')
            if type(self.pcd) is list:
                clean_pcd = []
                for pc in self.pcd:
                    cl, ind = pc.remove_radius_outlier(nb_points, radius)
                    cleaned_pcd = pc.select_by_index(ind)
                    clean_pcd.append(cleaned_pcd)
                self.pcd = clean_pcd

            else:
                cl, ind = self.pcd.remove_radius_outlier(nb_points, radius)
                cleaned_pcd = self.pcd.select_by_index(ind)
                self.pcd = cleaned_pcd

            self._save_ply("radius_filtered")

        else:
            print("No point cloud data to filter the outliers.")

    def remove_outliers_normal(self, radius=0.1, threshold_angle=30.0, max_nn=30):
        """
        Filter points whose normals deviate significantly from their local neighborhood normals.

        :param radius: Radius for local neighborhood search.
        :param threshold_angle: Angular threshold (in degrees) for filtering points based on normals.
        :param max_nn: Maximum number of neighbors to consider in the local search.
        :return: Filtered point cloud.
        """
        if self.pcd:
            pc = self.pcd[0]  # Assuming self.pcd contains a list of point clouds

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
                if k < 3:
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
            o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)

            # Update the point cloud with the filtered one
            self.pcd = filtered_pcd

        else:
            print("No point cloud data to filter the outliers.")

    def cluster_kmeans_normals(self, max_k=10, remove_ground=True, upward_threshold=0.5):
        """
        Cluster the point cloud based on normals using KMeans and the elbow method.
        """
        if self.pcd:
            if type(self.pcd) is list:
                # If the list contains 1 point cloud, get this pc and move on
                if len(self.pcd) == 1:
                    self.pcd = self.pcd[0]
                else:
                    print('use DBSCAN clustering and filtering to end up with only one point cloud')

            # Access the normals
            # print("Estimating normals...")
            # self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # # normals = self.pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
            # o3d.visualization.draw_geometries([self.pcd], point_show_normal=True)

            # Ensure that the point cloud has normals computed
            if not self.pcd.has_normals():
                self.estimate_normals()

            normals = np.asarray(self.pcd.normals)

            # Automatically choose the optimal number of clusters
            n_clusters = elbow_method(normals, self.output_dir, max_k=max_k)
            print(f"Amount of clusters: {n_clusters}")

            # Apply KMeans clustering
            clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

            # Get cluster labels
            labels = clustering.labels_

            # Separate points based on the clusters
            max_label = labels.max()
            clusters = []
            for i in range(max_label + 1):
                indices = np.where(labels == i)[0]
                cluster = self.pcd.select_by_index(indices)
                clusters.append(cluster)
                print(f"Cluster {i} has {len(cluster.points)} points.")

            if remove_ground:
                print("Filtering clusters that don't have normals pointing upwards...")
                non_upward_clusters = []
                for i, cluster in enumerate(clusters):
                    # Calculate the mean normal of the cluster
                    mean_normal = np.mean(np.asarray(cluster.normals), axis=0)

                    if mean_normal[2] <= upward_threshold:  # Z-component threshold
                        non_upward_clusters.append(cluster)

                clusters = non_upward_clusters  # Update clusters to only keep non-upward clusters
                print(f"Number of clusters after filtering: {len(clusters)}")

            self.pcd = clusters
            self._save_ply('cluster_kmeans')

        else:
            print("No point cloud data for clustering.")

    def cluster_dbscan(self, eps, min_samples, remove_small_clusters=True, min_points=20):
        """
        Cluster the point cloud using DBSCAN, which does not require specifying the number of clusters.

        Parameters:
        - eps (float): Maximum distance between two points to be considered in the same cluster.
        - min_samples (int): Minimum number of points in a neighborhood to form a core point.
        - remove_small_clusters (bool): Whether to remove small clusters.
        """
        if self.pcd:
            # Access the XYZ coordinates
            points = np.asarray(self.pcd.points)

            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

            # Get cluster labels
            labels = clustering.labels_
            max_label = labels.max()
            clusters = []

            for i in range(max_label + 1):
                indices = np.where(labels == i)[0]
                cluster = self.pcd.select_by_index(indices)
                clusters.append(cluster)
                print(f"Cluster {i} has {len(cluster.points)} points.")

            if remove_small_clusters:
                print(f"Number of clusters before filtering: {len(clusters)}")
                # Initialize an empty list to store valid clusters
                filtered_clusters = []
                for cluster in clusters:
                    if len(cluster.points) > min_points:
                        filtered_clusters.append(cluster)

                # Update the clusters list with only the filtered clusters
                clusters = filtered_clusters
                print(f"Number of clusters after filtering: {len(clusters)}")

            self.pcd = clusters
            self._save_ply('cluster_DBSCAN')

        else:
            print("No point cloud data for clustering.")

    def _save_ply(self, file_name, point_cloud=None):
        """
        Save the point cloud(s) with the given file name to PLY files.

        Parameters:
        file_name_prefix (str): The prefix for each PLY file name.
        point_cloud (list or PointCloud): A PointCloud or list of PointCloud clusters to be saved. If None, saves the main point cloud.
        """

        if point_cloud is None:
            point_cloud = self.pcd  # If no point cloud is provided, save the main point cloud

        if point_cloud:
            if type(point_cloud) is list:
                for i, pc in enumerate(point_cloud):
                    save_path = os.path.join(self.output_dir, f"{file_name}_{i}.ply")
                    o3d.io.write_point_cloud(save_path, pc)
            else:
                save_path = os.path.join(self.output_dir, f"{file_name}.ply")
                o3d.io.write_point_cloud(save_path, point_cloud)

        else:
            print("No point cloud(s) to save.")


class Mesh:
    def __init__(self, file_dir_mesh, file_name_mesh_list):
        """
        Initialize the Mesh class with the given directory and list of mesh filenames.
        Automatically find the latest 'ProgressPilot' output directory to store results.
        """
        self.file_dir_mesh = file_dir_mesh
        self.file_name_mesh_list = file_name_mesh_list  # List of mesh filenames
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
            print(f"Using the latest output directory: {self.output_dir}")
        else:
            raise Exception(f"No directories found in {main_dir}. Ensure that you first create a PointCloud project.")

    def load_meshes(self):
        """
        Load multiple meshes from the provided list of .ply files and store them in a list.
        """
        print(f"Loading mesh files from {self.file_dir_mesh}")
        for file_name in self.file_name_mesh_list:
            file_path = os.path.join(self.file_dir_mesh, file_name)
            if os.path.exists(file_path):
                print(f"Loading mesh: {file_path}")
                mesh = pv.read(str(file_path))

                self.meshes.append(mesh)
            else:
                print(f"File {file_name} does not exist in the directory {self.file_dir_mesh}.")

        self._save_meshes('model_mesh')

        return self.meshes

    def visualize(self, save_as_png=False, filename='mesh_visualization'):
        """
        Visualize the loaded meshes and optionally save the visualization as a PNG file.
        """
        if self.meshes:
            plotter = pv.Plotter()

            for mesh in self.meshes:
                # Generate a random color for each mesh
                color = np.random.rand(3)  # Random color [R, G, B]
                plotter.add_mesh(mesh, color=color)

            plotter.show()

            if save_as_png:
                filename = f'{filename}.png'
                plotter.screenshot(filename)
                print(f"Mesh visualization saved as {filename}")

        else:
            print("No meshes loaded to visualize.")

    def visualize_normals(self):
        """
        Load multiple PLY files, estimate normals if not present, and visualize each mesh and its normals.

        :param self.meshes: List of paths to PLY files.
        """
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
                cone = pv.Cone(center=point, direction=normal, radius=0.01,
                               height=0.1)  # Adjust radius and height for visibility
                plotter.add_mesh(cone, color='red')

        # Step 7: Show the plot
        plotter.show()

    def _save_meshes(self, file_name="mesh"):
        """
        Save all loaded meshes to the latest output directory with a given filename prefix.
        """
        if self.meshes:
            for i, mesh in enumerate(self.meshes):
                save_path = os.path.join(self.output_dir, f"{file_name}_{i}.ply")
                mesh.save(save_path)
                print(f"Saved mesh {i} as {save_path}")
        else:
            print("No meshes to save.")


class ComparePCDMesh:
    def __init__(self, point_clouds, meshes):
        """
        Initialize the Compare class with already loaded point clouds and meshes.

        :param point_clouds: A list of PointClouds.
        :param meshes: A list of Meshes.
        """
        self.pcd = point_clouds
        self.meshes = meshes

    def rotate_pcd(self):
        if self.pcd and self.meshes:
            # Compute mean normal in one point cloud cluster
            if type(self.pcd) is not list:
                self.pcd = list(self.pcd)

            pc = self.pcd[0]
            if not pc.has_normals():
                print('no normals yes')
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

            normals = np.asarray(pc.normals)
            normal_source = np.mean(normals, axis=0)
            print(f'normal source: {normal_source}')

            # Compute mean normal in one model mesh
            if type(self.meshes) is not list:
                self.meshes = list(self.meshes)

            mesh = self.meshes[0]
            normals_mesh = mesh.point_data['Normals']
            normal_target = np.mean(normals_mesh, axis=0)
            print(f'normal target: {normal_target}')

            rot_matrix = compute_rotation_matrix(normal_source, normal_target)
            print(f"Rotation Matrix:\n {rot_matrix}")

            # Perform rotation on the points and normals of all clusters
            rotated = []
            mean_normals_pcd = []
            for pc in self.pcd:
                points = np.asarray(pc.points)
                normals = np.asarray(pc.normals)

                rotated_points = points @ rot_matrix.T
                rotated_normals = normals @ rot_matrix.T

                # Update the point cloud with rotated points and normals
                pc.points = o3d.utility.Vector3dVector(rotated_points)
                pc.normals = o3d.utility.Vector3dVector(rotated_normals)

                rot_normals = np.asarray(pc.normals)
                mean_normal = np.mean(rot_normals, axis=0)
                mean_normal /= np.linalg.norm(mean_normal)
                mean_normals_pcd.append(mean_normal)
            print(f'mean normal pcd: {mean_normals_pcd}')
            self.visualize()

            # Compute mean normals of all mesh clusters
            mean_normals_meshes = []
            for mesh in self.meshes:
                normals = mesh.point_data['Normals']
                mean_normal = np.mean(normals, axis=0)
                mean_normal /= np.linalg.norm(mean_normal)
                mean_normals_meshes.append(mean_normal)
            print(f'mean normal mesh: {mean_normals_meshes}')

            # Check for matching normals
            for i, normal_pc in enumerate(mean_normals_pcd):
                for j, normal_mesh in enumerate(mean_normals_meshes):
                    alignment = np.dot(normal_pc, normal_mesh)
                    if alignment > 0.9:  # Threshold for similarity (adjust as needed)
                        print(f"Cluster {i} matches the mesh {j} with an alignment of {alignment:.2f}")
                        break
                    else:
                        continue

            # Visualize the result for the user
            # self.visualize()

            # Ask user to confirm the alignment
            # user_input = input(f"Is the alignment of cluster {i} correct? (yes/no): ")
            # if user_input.lower() == "yes":
            #     print("Alignment confirmed.")
            # else:
            #     print("Alignment rejected. Adjust the rotation.")
            #
            # for i in range(1, len(self.pcd)):
            #     # Check if the other normals are the same now


            # while True:
            #     self.visualize()
            #
            #     rot_feedback = input("Is the rotation correct?")
            #     if rot_feedback == 'yes' or rot_feedback == 'Yes':
            #         break
            #     else:
            #         pass


            # while True:
            #     self.visualize()
            #
            #     rot_feedback = input("Is the rotation correct?")
            #     if rot_feedback == 'yes' or rot_feedback == 'Yes':
            #         break
            #     else:
            #         rot_adjustment = input("How does it need to be rotated? [90 left, 90 right, 180]")
            #         if rot_adjustment == '90 left':
            #             # Anti-clockwise rotation (90 degrees around Z-axis)
            #             rotation_matrix_new = np.array([[0, -1, 0],
            #                                             [1, 0, 0],
            #                                             [0, 0, 1]])
            #         elif rot_adjustment == '90 right':
            #             # Clockwise rotation (90 degrees around Z-axis)
            #             rotation_matrix_new = np.array([[0, 1, 0],
            #                                             [-1, 0, 0],
            #                                             [0, 0, 1]])
            #         elif rot_adjustment == '180':
            #             # 180-degree rotation around Z-axis
            #             rotation_matrix_new = np.array([[-1, 0, 0],
            #                                             [0, -1, 0],
            #                                             [0, 0, 1]])
            #         else:
            #             raise ValueError("Direction must be '90 left', '90 right', or '180'.")

            # for pc in self.pcd:
            #     points = np.asarray(pc.points)
            #     normals = np.asarray(pc.normals)
            #
            #     rotated_points = points @ rotation_matrix_new.T
            #     rotated_normals = normals @ rotation_matrix_new.T
            #
            #     # Update the point cloud with rotated points and normals
            #     pc.points = o3d.utility.Vector3dVector(rotated_points)
            #     pc.normals = o3d.utility.Vector3dVector(rotated_normals)
            #
            #     rotated.append(pc)

            self.pcd = rotated

    def translate_pcd(self):

        if self.pcd and self.meshes:
            # Find minimum corner point
            min_x_values = []
            min_y_values = []
            min_z_values = []
            for pc in self.pcd:
                min_point = pc.get_min_bound()

                min_x_values.append(min_point[0])
                min_y_values.append(min_point[1])
                min_z_values.append(min_point[2])

            min_x = np.min(min_x_values)
            min_y = np.min(min_y_values)
            min_z = np.min(min_z_values)

            # New corner point
            corner_point = np.array([min_x, min_y, min_z])
            print(f'min bound: {corner_point}')

            # Translate both point clouds so the new corner point is at the origin
            translation_vector = -corner_point
            print(f'Translation vector: {translation_vector}')

            # Perform translation on the points and normals of all clusters
            translated = []
            for pc in self.pcd:
                points = np.asarray(pc.points)
                normals = np.asarray(pc.normals)

                translated_points = points + translation_vector
                translated_normals = normals + translation_vector

                # Update the point cloud with rotated points and normals
                pc.points = o3d.utility.Vector3dVector(translated_points)
                pc.normals = o3d.utility.Vector3dVector(translated_normals)

                translated.append(pc)

            self.pcd = translated


    def visualize(self, save_as_png=False, filename='compare_visualization'):
        if self.pcd:
            plotter = pv.Plotter()

            # Add meshes
            for mesh in self.meshes:
                plotter.add_mesh(mesh, color='lightgrey')

            # Add point clouds
            for pc in self.pcd:
                points_pc = np.asarray(pc.points)
                plotter.add_mesh(points_pc, color='lightblue')

            plotter.show()

            if save_as_png:
                filename = f'{filename}.png'
                plotter.screenshot(filename)
                print(f"Point cloud and mesh visualization saved as {filename}")

        else:
            print("No point clouds loaded to visualize.")