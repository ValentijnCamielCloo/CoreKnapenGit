from constants import FILE_NAME_MESH_LIST
from functions import *
import constants as c

def main():
    # Create an instance of the point cloud
    pcd = PointCloud(file_name_pcd=c.FILE_NAME_PCD)

    # Load the point cloud
    pcd.load_pcd()
    # pcd.visualize()

    # Downsample the point cloudq
    pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    pcd.estimate_normals(orientate_camera=True)

    # Cluster point cloud based on normals
    pcd.cluster_kmeans_normals()
    pcd.visualize()

    # Remove outliers from the clusters
    pcd.remove_outliers_radius(nb_points=c.NB_POINTS, radius=c.RADIUS)
    pcd.visualize()

    # Initialize the Mesh class with the directory and list of files
    meshes = Mesh(file_dir_mesh=c.FILE_DIR_MESH, file_name_mesh_list=FILE_NAME_MESH_LIST)

    # Load the meshes
    meshes.load_meshes()
    meshes.visualize(title='Mesh')

    # Rotate the point cloud based on normal
    pcd.orientate(meshes.meshes)
    # #
    # Check if the orientation is correct
    # compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    # compare.visualize(title='Check orientation point cloud')
    #
    # # Register the point clouds to one point cloud
    # # pcd.registration(source_pcd=c.SOURCE_PCD)
    # # pcd.visualize()
    #
    # # Cluster dbscan
    # pcd.cluster_dbscan(eps=c.EPS, min_samples=c.MIN_SAMPLES)
    # pcd.visualize()
    #
    # # Cluster k means the planes
    # pcd.cluster_kmeans_normals()
    # pcd.visualize()
    #
    # Filter the total point cloud
    # pcd.estimate_normals()
    # pcd.remove_outliers_radius(nb_points=c.NB_POINTS,radius=c.RADIUS)
    # pcd.remove_outliers_normal(radius=c.RADIUS, threshold_angle=c.THRESHOLD_ANGLE, max_nn=c.MAX_NN)
    # pcd.visualize()

    # # Cluster point cloud based on normals
    # # pcd.cluster_kmeans_normals()
    # # pcd.visualize()
    # #
    # Translate point cloud to the origin (0,0)
    pcd.translate()
    # #
    # # Check if the normals are pointing to the outside
    # meshes.visualize_normals()
    #
    # Initialize the ComparePCDMesh class
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    compare.visualize(title='Check pcd and mesh')
    compare.check_bricks(points_per_brick=c.POINTS_PER_BRICK)
    compare.visualize_result(filename_vis=c.FILENAME_VIS)
    #
    compare.write_results()


if __name__ == '__main__':
    main()