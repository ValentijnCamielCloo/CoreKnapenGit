import open3d as o3d    #Version 0.18.0
import numpy as np      #Version 1.26.4
import copy
import os
from xx_functions_registration import *

# Define the folder path where your PLY files are stored
folder_path = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\out_01-10' #AANPASSEN

# Specify the PLY file names
ply_file_source = os.path.join(folder_path, "point_cloud_20241001_132244.ply") #AANPASSEN
ply_file_target = os.path.join(folder_path, "point_cloud_20241001_132325.ply") #AANPASSEN

# Read the point clouds with Open3D
source_pcd = o3d.io.read_point_cloud(ply_file_source)
target_pcd = o3d.io.read_point_cloud(ply_file_target)

# Paint the point clouds with different colors
source_pcd.paint_uniform_color([1, 0.706, 0])  # Orange for source
target_pcd.paint_uniform_color([0, 0.651, 0.929])  # Blue for target
# o3d.visualization.draw_geometries([source_pcd, target_pcd])

# Downsample the point cloud with Voxel Downsampling
voxel_size = 0.01
source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

# o3d.visualization.draw_geometries([source_down])

#Outlier removal
print("Statistical oulier removal: Source")
cl, ind = source_down.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.5)
# display_inlier_outlier(source_down, ind)
source_inlier_cloud = source_down.select_by_index(ind)
# o3d.visualization.draw_geometries([source_inlier_cloud])
point = np.asarray(source_inlier_cloud.points)
print("Points:", point)

print("Statistical oulier removal: Target")
cl, ind = target_down.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.5)
# display_inlier_outlier(target_down, ind)
target_inlier_cloud = target_down.select_by_index(ind)
# o3d.visualization.draw_geometries([target_inlier_cloud])

# o3d.visualization.draw_geometries([source_down, target_down])

# Global registration to get the initial alignment
trans_global = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

# draw_registration_result(source_down, target_down, trans_global.transformation)

# Local registration for fine-tuning alignment (ICP)
threshold = 0.05
trans_local = execute_local_registration(source_down, target_down,
                                         voxel_size, trans_global)

# draw_registration_result(source_down, target_down, trans_local.transformation)