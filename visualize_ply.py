import open3d as o3d    #Version 0.18.0
import numpy as np      #Version 1.26.4
import os

# Define the folder path where your PLY files are stored
folder_path = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\scans'

# Specify the PLY file names
ply_file_1 = os.path.join(folder_path, "Scan_46_20241015_161510_filtered.ply")
pcd = o3d.io.read_point_cloud(ply_file_1)

o3d.visualization.draw_geometries([pcd])
