import os
import re
from datetime import datetime
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Folder where scans are stored
folder_path = "output"

# Define regex pattern for scan filenames: Scan_number_date_time_filtered.ply
pattern = re.compile(r"Scan_(\d+)_(\d{8})_(\d{6})_filtered\.ply")

def get_latest_scan(folder_path):
    latest_scan = None
    latest_datetime = None

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            scan_date = match.group(2)  # e.g., '20241011'
            scan_time = match.group(3)  # e.g., '110602'
            # Combine date and time into a single datetime object
            scan_datetime = datetime.strptime(scan_date + scan_time, "%Y%m%d%H%M%S")

            # Find the most recent scan
            if latest_datetime is None or scan_datetime > latest_datetime:
                latest_datetime = scan_datetime
                latest_scan = filename

    return latest_scan

# Get the latest scan file
latest_scan_file = get_latest_scan(folder_path)
if latest_scan_file:
    print(f"Latest scan file: {latest_scan_file}")
    ply_path = os.path.join(folder_path, latest_scan_file)
    
    # Load the point cloud from the latest scan
    pcd = o3d.io.read_point_cloud(ply_path)

    # Optional: Downsample the point cloud for faster processing
    pcd_down = pcd.voxel_down_sample(voxel_size=0.005)

    # Plane segmentation using RANSAC to detect and remove the floor
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01,
                                                  ransac_n=3,
                                                  num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Filter out the floor points
    pcd_without_floor = pcd_down.select_by_index(inliers, invert=True)

    # Compute normals for edge detection
    pcd_without_floor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01, max_nn=30))

    # Optional: Orient the normals consistently
    pcd_without_floor.orient_normals_consistent_tangent_plane(k=50)

    # Perform clustering for segmentation
    # Adjusted DBSCAN clustering
    eps_value = 0.005  # Reducing epsilon to try to capture more clusters
    min_points_value = 5  # Lower minimum points for smaller clusters

    # Perform clustering for segmentation
    labels = np.array(pcd_without_floor.cluster_dbscan(eps=eps_value, 
                                                    min_points=min_points_value, 
                                                    print_progress=True))

    # Visualize segmentation
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Unclustered points in black
    pcd_without_floor.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Visualize the result
    o3d.visualization.draw_geometries([pcd_without_floor])


else:
    print("No valid scan files found in the folder.")
