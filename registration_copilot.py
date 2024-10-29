import open3d as o3d
import os
import datetime

# Define the folder paths
scans_folder_path = r'D:\TUdelftGitCore\CoreKnapenGit\scans'
output_base_folder = r'D:\TUdelftGitCore\CoreKnapenGit\ProgressPilotRegistration'

# Create a folder for the current run with date and time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(output_base_folder, f"ProgressPilotRegistration_{current_time}")
os.makedirs(output_folder, exist_ok=True)

# Get all PLY files from the scans folder
ply_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith(".ply")])

# Initialize a list to hold all the point clouds
point_clouds = []

# Load all point clouds and preprocess them
for ply_file in ply_files:
    pcd = o3d.io.read_point_cloud(os.path.join(scans_folder_path, ply_file))
    pcd = pcd.voxel_down_sample(voxel_size=0.01)  # Use a smaller voxel size
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_clouds.append(pcd)

# Visualize the initial point clouds
for i, pcd in enumerate(point_clouds):
    print(f"Visualizing point cloud {i+1}")
    o3d.visualization.draw_geometries([pcd])

# Now we have a list of all point clouds
# Proceed with feature extraction, initial alignment, and fine registration

# For demonstration, let's take the first two point clouds and register them
pcd1 = point_clouds[0]
pcd2 = point_clouds[1]

# Feature extraction
fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
    pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
    pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Initial alignment
distance_threshold = 0.05
checker = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]

init_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    pcd1, pcd2, fpfh1, fpfh2, mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3, 
    checkers=checker,
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

# Visualize the initial alignment
print("Visualizing initial alignment")
o3d.visualization.draw_geometries([pcd1, pcd2])

# Fine registration
icp_result = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, 0.02, init_result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

# Transform point cloud
pcd1.transform(icp_result.transformation)

# Visualize the fine alignment
print("Visualizing fine alignment")
o3d.visualization.draw_geometries([pcd1, pcd2])

# Merge point clouds
combined_pcd = pcd1 + pcd2
# Add more point clouds here as needed

# Save the merged point cloud
o3d.io.write_point_cloud(os.path.join(output_folder, "merged_cloud.ply"), combined_pcd)

# Visualize the final merged point cloud
print("Visualizing merged point cloud")
o3d.visualization.draw_geometries([combined_pcd])
