import open3d as o3d  # Version 0.18.0
import numpy as np  # Version 1.26.4
import copy

# Function to visualize the registration
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Paint the point clouds with different colors
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue for target

    # Apply the transformation to the source point cloud
    source_temp.transform(transformation)

    # Use Open3D's built-in visualization to display the point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp])

#Function for downsampling the point cloud
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (black): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    reg_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print("global registration: ", reg_global)
    return reg_global

def execute_local_registration(source_down, target_down, voxel_size, trans_init, type="PointToPlane"):
    # Smaller threshold will make it more accurate
    threshold = voxel_size * 0.5
    if type == "PointToPlane":
        print("Apply point-to-plane ICP")
        # Registration Point to Plane:registration::TransformationEstimationPointToPlane
        reg_local = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        print("Apply point-to-point ICP")
        # Registration Point to Plane:registration::TransformationEstimationPointToPoint
        reg_local = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print("local registration: ", reg_local)
    return reg_local

def evaluation_registration(source_down, target_down, threshold, result_reg):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, threshold, result_reg.transformation)
    print("Evaluation: ",evaluation)