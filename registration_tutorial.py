import os
import re
from datetime import datetime
import open3d as o3d
import numpy as np
from tqdm import tqdm  # For progress bar

# Folder where scans are stored
folder_path = "output"
merged_folder = "merged"

# Create the merged folder if it doesn't exist
if not os.path.exists(merged_folder):
    os.makedirs(merged_folder)

# Define regex pattern for scan filenames: Scan_number_date_time_filtered.ply
pattern = re.compile(r"Scan_(\d+)_(\d{8})_(\d{6})_filtered\.ply")

def get_latest_scans(folder_path, n=2):
    scans = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            scan_date = match.group(2)  # e.g., '20241011'
            scan_time = match.group(3)  # e.g., '110602'
            # Combine date and time into a single datetime object
            scan_datetime = datetime.strptime(scan_date + scan_time, "%Y%m%d%H%M%S")
            scans.append((filename, scan_datetime))

    # Sort scans by datetime and return the latest n files
    scans_sorted = sorted(scans, key=lambda x: x[1], reverse=True)
    return [scan[0] for scan in scans_sorted[:n]]

# Get the 4 most recent filtered .ply files
latest_scans = get_latest_scans(folder_path)
print(f"Latest scans: {latest_scans}")

# Load the point clouds
def load_point_clouds(folder_path, scan_files):
    point_clouds = []
    for scan_file in scan_files:
        pcd = o3d.io.read_point_cloud(os.path.join(folder_path, scan_file))
        point_clouds.append(pcd)
    return point_clouds

# Load the 4 most recent point clouds
point_clouds = load_point_clouds(folder_path, latest_scans)

# Function to perform pairwise registration using ICP with progress output
def pairwise_registration(source, target, threshold=0.02):
    print(f"Registering point clouds...")

    # Display a progress bar for the registration process
    iterations = 2000  # Number of iterations for ICP
    for i in tqdm(range(iterations), desc="ICP Progress"):
        # Perform ICP registration step
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i+1)
        )
    return icp_result.transformation

# Merge the point clouds
def merge_point_clouds(point_clouds):
    base_pcd = point_clouds[0]

    for i in range(1, len(point_clouds)):
        source_pcd = point_clouds[i]
        transformation = pairwise_registration(source_pcd, base_pcd)
        source_pcd.transform(transformation)
        base_pcd += source_pcd

    # Optional: Downsample the merged point cloud
    base_pcd = base_pcd.voxel_down_sample(voxel_size=0.005)
    return base_pcd

# Merge the point clouds
merged_pcd = merge_point_clouds(point_clouds)

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Point Cloud")

# Extract scan number from the latest scan file and increment it for the merged file
latest_scan = latest_scans[0]
match = pattern.match(latest_scan)
if match:
    scan_number = int(match.group(1)) + 1  # Increment the scan number by 1
    scan_date = match.group(2)
    scan_time = match.group(3)

# Create the filename in the format Scan_number_date_time_merged.ply
merged_filename = f"Scan_{scan_number}_{scan_date}_{scan_time}_merged.ply"
merged_filepath = os.path.join(merged_folder, merged_filename)

# Save the merged (registered) point cloud
o3d.io.write_point_cloud(merged_filepath, merged_pcd)
print(f"Merged point cloud saved to {merged_filepath}")
