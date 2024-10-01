import open3d as o3d    #Version 0.18.0
import numpy as np      #Version 1.26.4
import copy
import os
from functions_registration import *

# Define the folder path where your PLY files are stored
folder_path = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\out_01-10'

# Specify the PLY file names
ply_file_source = os.path.join(folder_path, "point_cloud_20241001_132244.ply")
ply_file_target = os.path.join(folder_path, "point_cloud_20241001_132325.ply")

# Read the point clouds with Open3D
source_pcd = o3d.io.read_point_cloud(ply_file_source)
target_pcd = o3d.io.read_point_cloud(ply_file_target)

# Paint the point clouds with different colors
source_pcd.paint_uniform_color([1, 0.706, 0])  # Orange for source
target_pcd.paint_uniform_color([0, 0.651, 0.929])  # Blue for target
o3d.visualization.draw_geometries([source_pcd, target_pcd])

# Downsample the point cloud with Voxel Downsampling
voxel_size = 0.05
source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

# o3d.visualization.draw_geometries([source_down])

# Global registration to get the initial alignment
trans_global = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

draw_registration_result(source_pcd, target_pcd, trans_global.transformation)

# Local registration for fine-tuning alignment (ICP)
threshold = 0.05
trans_local = execute_local_registration(source_down, target_down,
                                         voxel_size, trans_global)

draw_registration_result(source_pcd, target_pcd, trans_local.transformation)