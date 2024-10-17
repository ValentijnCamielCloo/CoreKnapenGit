import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime
import pyvista as pv


def elbow_method(normals, max_k=10):
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
    plt.savefig('elbow_plot', format='png', dpi=300)
    print(f"Elbow plot saved as elbow_plot.png")
    plt.show()

    return optimal_k


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

    def visualize(self):
        """
        Visualize the current point cloud.
        """
        if self.pcd is not None:
            print("Visualizing point cloud.")
            o3d.visualization.draw_geometries([self.pcd])
        else:
            print("No point cloud data to visualize.")

    def voxel_downsample(self, voxel_size):
        """
        Downsample the point cloud using a voxel grid filter.

        Parameters:
        voxel_size (float): The size of the voxel grid filter.
        """
        if self.pcd is not None:
            print(f"Downsampling point cloud with voxel size {voxel_size}")
            self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
            self._save_ply("downsampled")
        else:
            print("No point cloud data to downsample.")

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

    def cluster_based_on_normals(self, max_k=10):
        """
        Cluster the point cloud based on normals using KMeans and the elbow method.
        """
        if self.pcd is not None:
            # Step 1: Access the normals
            normals = np.asarray(self.pcd.normals)

            # Step 2: Automatically choose the optimal number of clusters
            n_clusters = elbow_method(normals, max_k=max_k)
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

            return clusters
        else:
            print("No point cloud data for clustering.")
            return []

    def filter_clusters(self, clusters, upward_threshold=0.5):
        """
        Filter clusters based on the Z-component of their mean normals.
        """
        print("Filtering clusters that don't have normals pointing upwards...")
        non_upward_clusters = []
        for i, cluster in enumerate(clusters):
            # Calculate the mean normal of the cluster
            mean_normal = np.mean(np.asarray(cluster.normals), axis=0)
            print(f"Cluster {i}: mean normal = {mean_normal}")

            if mean_normal[2] <= upward_threshold:  # Z-component threshold
                non_upward_clusters.append(cluster)

        output_folder = rf"{self.file_path}\segmentation"
        os.makedirs(output_folder, exist_ok=True)
        for i, cluster in enumerate(non_upward_clusters):
            self._save_ply(rf"{output_folder}/cluster_{i}.ply", cluster)

        return non_upward_clusters

    def _save_ply(self, file_name, point_cloud=None):
        """
        Save the point cloud with the given file name to a PLY file.
        """
        if point_cloud is None:
            point_cloud = self.pcd

        if point_cloud is not None:
            save_path = os.path.join(self.output_dir, f"{file_name}.ply")
            o3d.io.write_point_cloud(save_path, point_cloud)
            print(f"Point cloud saved as {save_path}")