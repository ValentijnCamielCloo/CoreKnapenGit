from functions import *
import constants as c

def main():
    # Create an instance of the point cloud
    pcd = PointCloud(file_dir_pcd=c.FILE_DIR_PCD, file_name_pcd=c.FILE_NAME_PCD)

    # Load and process the point cloud
    pcd.load_pcd()
    pcd.visualize()
    # pcd.estimate_and_orient_normals()
    #
    # # Cluster based on normals
    # clusters = pcd.cluster_based_on_normals(max_k=10)
    #
    # # Filter non-upward clusters
    # non_upward_clusters = pcd.filter_non_upward_clusters(clusters)
    #
    # # Save and visualize the non-upward clusters
    # pcd.save_clusters(non_upward_clusters)
    # pcd.visualize_clusters(non_upward_clusters)

if __name__ == '__main__':
    main()