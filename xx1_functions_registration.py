import os
import numpy as np
import open3d as o3d
from datetime import datetime
#xxXX
class PointCloudProcessor:
    def __init__(self, pcd=None, output_dir="D:/TUdelftGitCore/CoreKnapenGit/ProgressPilotRegistration", filenames=None):
        self.pcd = [] if pcd is None else pcd  # Ensure pcd is a list
        self.base_output_dir = output_dir
        self.output_dir = self._create_new_output_directory()
        self.filenames = filenames
        self.step_counter = 1  # Initialize step counter

    def _create_new_output_directory(self):
        """ Create a new directory for each run based on date and time. """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_name = f"ProgressPilotRegistration_{timestamp}"
        new_dir_path = os.path.join(self.base_output_dir, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path

    def voxel_downsample(self, voxel_size):
        """ Downsample all point clouds using a voxel grid filter. """
        if self.pcd:
            print(f"Downsampling point clouds with voxel size {voxel_size}")
            self.pcd = [pc.voxel_down_sample(voxel_size=voxel_size) for pc in self.pcd]
            self._save_ply("downsampled")
        else:
            print("No point cloud data to downsample.")

    def remove_outliers_radius(self, nb_points, radius):
        """ Remove outliers from clusters using radius outlier removal. """
        if self.pcd:
            print('Removing outliers...')
            clean_pcd = []
            for pc in self.pcd:
                cl, ind = pc.remove_radius_outlier(nb_points, radius)
                cleaned_pcd = pc.select_by_index(ind)  # Select indices from the current point cloud
                clean_pcd.append(cleaned_pcd)
            self.pcd = clean_pcd

            self._save_ply("radius_filtered")
        else:
            print("No point cloud data to filter the outliers.")

    def _save_ply(self, suffix):
        """ Save point clouds as PLY files in the specified output directory. """
        if isinstance(self.pcd, list):
            for i, pc in enumerate(self.pcd):
                prefix = f"{self.step_counter}_ProgressPilotRegistration_{suffix}"
                filename = f"{self.output_dir}/{prefix}_{'Target' if i == 1 else 'Source'}.ply"
                o3d.io.write_point_cloud(filename, pc)
                print(f"Saved {filename}")

        self.step_counter += 1  # Increment step counter
