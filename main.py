from functions import *
import constants as c

def main():
    # Create an instance of the point cloud
    pcd = PointCloud(file_dir_pcd=c.FILE_DIR_PCD, file_name_pcd=c.FILE_NAME_PCD)

    # Load the point cloud
    pcd.load_pcd()
    # pcd.visualize()

    # Downsample the point cloud
    pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    # pcd.visualize()
    # pcd.visualize(save_as_png=True, filename='downsampled')

    # Cluster point cloud based on normals
    pcd.estimate_and_orient_normals()
    pcd.cluster_based_on_normals()
    # pcd.visualize()

    # Remove outliers from the clusters
    pcd.remove_outliers_radius(nb_points=c.NB_POINTS, radius=c.RADIUS)
    pcd.visualize()

    # Calculate mean normal vectors for each cluster
    mean_normals = pcd.calculate_mean_normal_vector()




if __name__ == '__main__':
    main()