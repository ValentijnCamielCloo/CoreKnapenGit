import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
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


def visualize_clusters(clusters, color=True):
    """
    Visualize the clusters in the point cloud.

    Parameters:
    clusters (list): A list of clustered point clouds to visualize.
    color (bool): Whether to randomly color each cluster.
    """
    if clusters:
        plotter = pv.Plotter()
        for i, cluster in enumerate(clusters):
            pcd_points = np.asarray(cluster.points)
            # Create a PyVista PolyData object for visualization
            cloud = pv.PolyData(pcd_points)

            if color:
                colors = np.random.rand(len(pcd_points), 3)  # Random RGB colors for each cluster
                plotter.add_mesh(cloud, color=colors[i % len(colors)])  # Assign color to the cluster
            else:
                plotter.add_mesh(cloud)

        plotter.show()

    else:
        print("No cluster data to visualize.")


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

    def estimate_and_orient_normals(self, radius=0.1, max_nn=30, camera_location=[0, 0, 0]):
        """
        Estimate and orient normals for the point cloud.
        """
        if self.pcd is not None:
            print("Estimating normals...")
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
            normals = self.pcd.orient_normals_towards_camera_location(camera_location=camera_location)
            return normals
        else:
            print("No point cloud data to estimate normals.")

    def cluster_based_on_normals(self, max_k=10, remove_ground=True, upward_threshold=0.5):
        """
        Cluster the point cloud based on normals using KMeans and the elbow method.
        """
        if self.pcd is not None:
            # Step 1: Access the normals
            normals = np.asarray(self.pcd.normals)

            # Step 2: Automatically choose the optimal number of clusters
            n_clusters = elbow_method(normals, self.output_dir, max_k=max_k)
            print(f"Amount of clusters: {n_clusters}")

            # Step 3: Apply KMeans clustering
            clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(normals)

            # Step 4: Get cluster labels
            labels = clustering.labels_

            # Step 5: Separate points based on the clusters
            max_label = labels.max()
            clusters = []
            for i in range(max_label + 1):
                indices = np.where(labels == i)[0]
                cluster = self.pcd.select_by_index(indices)
                clusters.append(cluster)
                print(f"Cluster {i} has {len(cluster.points)} points.")  # Debug: print number of points in cluster

            if remove_ground:
                print("Filtering clusters that don't have normals pointing upwards...")
                non_upward_clusters = []
                for i, cluster in enumerate(clusters):
                    # Calculate the mean normal of the cluster
                    mean_normal = np.mean(np.asarray(cluster.normals), axis=0)

                    if mean_normal[2] <= upward_threshold:  # Z-component threshold
                        non_upward_clusters.append(cluster)

                clusters = non_upward_clusters  # Update clusters to only keep non-upward clusters
                print(
                    f"Number of clusters after filtering: {len(clusters)}")  # Debug: print number of clusters after filtering

            # Save each cluster as a separate PLY file
            for i, cluster in enumerate(clusters):
                if len(cluster.points) > 0:  # Debug: check if the cluster has points
                    print(f"Saving cluster {i} with {len(cluster.points)} points.")  # Debug: print saving message
                    self._save_ply(f"cluster_{i}", point_cloud=cluster)
                else:
                    print(f"Cluster {i} is empty, skipping save.")  # Debug: skip empty clusters

            self.pcd = clusters

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
        Save the list of point clouds with the given file name to PLY files.

        Parameters:
        file_name_prefix (str): The prefix for each PLY file name.
        point_cloud (list): A list of point cloud clusters to be saved. If None, saves the main point cloud.
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



class ClustersPCD:
    def __init__(self, clusters):
        """
        Initialize the ClustersPCD with clustered point clouds.

        Parameters:
        clusters (list): A list of clustered point clouds to manipulate.
        """
        self.clusters = clusters  # Store the clusters in an instance variable

    def visualize(self, color=True):
        """
        Visualize the clusters in the point cloud.

        Parameters:
        color (bool): Whether to randomly color each cluster.
        """
        if self.clusters:
            plotter = pv.Plotter()
            for i, cluster in enumerate(self.clusters):
                pcd_points = np.asarray(cluster.points)
                # Create a PyVista PolyData object for visualization
                cloud = pv.PolyData(pcd_points)

                if color:
                    colors = np.random.rand(len(pcd_points), 3)  # Random RGB colors for each cluster
                    plotter.add_mesh(cloud, color=colors[i % len(colors)])  # Assign color to the cluster
                else:
                    plotter.add_mesh(cloud)

            plotter.show()
        else:
            print("No cluster data to visualize.")

    def remove_outliers(self, nb_points=12, radius=0.02):
        """
        Remove outliers from clusters using radius outlier removal.

        Parameters:
        nb_points (int):
        radius (float):
        """

        cleaned_clusters = []
        for cluster in self.clusters:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(cluster.points))

            # Outlier Removal
            cl, ind = pcd.remove_radius_outlier(nb_points, radius)
            cleaned_cluster = pcd.select_by_index(ind)
            cleaned_clusters.append(cleaned_cluster)

        self.clusters = cleaned_clusters  # Update clusters with cleaned data

    def calculate_mean_normal_vector(self):
        """
        Calculate the mean normal vector for each cluster.

        Returns:
        list: A list of mean normal vectors for each cluster.
        """
        mean_normals = []
        for i, cluster in enumerate(self.clusters):
            cluster.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.asarray(cluster.normals)
            mean_normal = np.mean(normals, axis=0)  # Compute mean normal
            mean_normals.append(mean_normal)
            print(f"Mean normal vector of cluster {i}: {mean_normals[i]}")
        return mean_normals

    def _save_ply(self, file_name, point_cloud=None):
        """
        Save the point cloud with the given file name to a PLY file.
        """
        if point_cloud is None:
            point_cloud = self.pcd

        if point_cloud is not None:
            save_path = os.path.join(self.output_dir, f"{file_name}.ply")
            o3d.io.write_point_cloud(save_path, point_cloud)
            # print(f"Point cloud saved as {save_path}")


# class Mesh:
