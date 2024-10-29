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
            print("Estimating normals...")
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = self.pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
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

    def calculate_mean_normal_vector(self):
        """
        Calculate the mean normal vector for each cluster.

        Returns:
        list: A list of mean normal vectors for each cluster.
        """
        if self.pcd:
            if type(self.pcd) is list:
                mean_normals = []
                for i, pc in enumerate(self.pcd):
                    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    normals = np.asarray(pc.normals)
                    mean_normal = np.mean(normals, axis=0)  # Compute mean normal
                    mean_normals.append(mean_normal)
                    print(f"Mean normal vector of cluster {i}: {mean_normals[i]}")
            else:
                self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                normals = np.asarray(self.pcd.normals)
                mean_normals = np.mean(normals, axis=0)  # Compute mean normal
                print(f"Mean normal vector of cluster: {mean_normals}")

            return mean_normals

        else:
            print("No point cloud data for calculating mean normal vector.")


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
        if self.pcd:

            # Compute mean normal in one point cloud cluster
            if type(self.pcd) is not list:
                print('was not a list')
                self.pcd = list(self.pcd)
            else:
                print('was already a list')

            pc = self.pcd[0]
            normals_pc = np.asarray(pc.normals)
            normal_source = np.mean(normals_pc, axis=0)

            # Compute mean normal in one model mesh
            if type(self.meshes) is not list:
                self.meshes = list(self.meshes)

            mesh = self.meshes[0]
            normals_mesh = mesh.point_data['Normals']
            normal_target = np.mean(normals_mesh, axis=0)

            rot_matrix = compute_rotation_matrix(normal_source, normal_target)
            print(f"Rotation Matrix:\n {rot_matrix}")

            for pc in self.pcd:
                points = np.asarray(pc.points)
                normals = np.asarray(pc.normals)

                rotated_points = points @ rot_matrix.T
                rotated_normals = normals @ rot_matrix.T

    def visualize(self):
        pass