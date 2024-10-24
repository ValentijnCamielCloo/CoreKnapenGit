
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
    pcd.cluster_dbscan(eps=c.EPS, min_samples=c.MIN_SAMPLES, min_points=c.MIN_POINTS)
    # pcd.visualize()
    # pcd.visualize(save_as_png=True, filename='downsampled')

    # Filter the point cloud
    pcd.estimate_normals()
    pcd.remove_outliers_radius(nb_points=c.NB_POINTS,radius=c.RADIUS)
    pcd.remove_outliers_normal()
    pcd.visualize()

    # Cluster point cloud based on normals
    pcd.cluster_kmeans_normals()
    pcd.visualize()

    # Initialize the Mesh class with the directory and list of files
    meshes = Mesh(file_dir_mesh=c.FILE_DIR_MESH, file_name_mesh_list=c.FILE_NAME_MESH_LIST)

    # Load the meshes
    meshes.load_meshes()
    # Check if the normals are pointing to the outside
    meshes.visualize_normals()

    # Initialize the ComparePCDMesh class
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)

    compare.rotate_pcd()
    compare.visualize()

    # compare.translate_pcd()
    # compare.visualize()


if __name__ == '__main__':
    main()