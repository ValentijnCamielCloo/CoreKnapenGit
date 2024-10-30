from matplotlib.pyplot import title

from functions import *
import constants as c

def main():
    # # Create an instance of the point cloud
    # pcd = PointCloud(file_name_pcd=c.FILE_NAME_PCD)
    #
    # # Load the point cloud
    # pcd.load_pcd()
    # pcd.visualize('scan', save_as_png=True)
    #
    # # Downsample the point cloudq
    # pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    # pcd.estimate_normals(orientate_camera=True)
    # pcd.visualize('downsampled', save_as_png=True)
    #
    # # Remove outliers
    # pcd.cluster_kmeans_normals(biggest_cluster=True)
    # pcd.remove_outliers_radius(nb_points=c.NB_POINTS,radius=c.RADIUS_RADIUS_REMOVAL)
    # pcd.visualize(title='Outliers removed', save_as_png=True)
    #
    # # Initialize the Mesh class with the directory and list of files
    # meshes = Mesh()
    # meshes.load_meshes()
    # meshes.visualize(title='Mesh', save_as_png=True)
    #
    # # Rotate the point cloud based on normal
    # pcd.orientate(meshes.meshes)

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
    # # Translate point cloud to the origin (0,0)
    # pcd.translate()
    # # #
    # # # Check if the normals are pointing to the outside
    # # meshes.visualize_normals()
    # #
    # # Initialize the ComparePCDMesh class
    # compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    # compare.visualize(title='Check pcd and mesh', save_as_png=True)
    # compare.check_bricks(points_per_brick=c.POINTS_PER_BRICK)
    # compare.visualize_result(filename_vis=c.FILENAME_VIS, save_as_png=True)
    # #
    # compare.write_results()

    pcd = PointCloud()
    pcd.initial_alignment()
    # pcd.estimate_normals(visualize_normals=True)
    pcd.visualize('initial alignment',save_as_png=True)

    # Initialize the Mesh class with the directory and list of files
    meshes = Mesh()
    meshes.load_meshes()

    # Translate point cloud to align the model
    pcd.translate(dist_scanner_obj=c.DIST_SCANNER_OBJ, height_scanner=c.HEIGHT_SCANNER)
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    compare.visualize(title='Check pcd and mesh', save_as_png=True)

    pcd.registration()
    pcd.visualize('Registered', save_as_png=True)

    # Downsample the point cloud
    pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    pcd.visualize('downsampled', save_as_png=True)
    pcd.estimate_normals(visualize_normals=True)

    # Remove outliers
    # pcd.remove_outliers_normal(radius=c.RADIUS_NORMAL_REMOVAL,threshold_angle=c.THRESHOLD_ANGLE,max_nn=c.MAX_NN)
    pcd.remove_outliers_radius(nb_points=c.NB_POINTS,radius=c.RADIUS_RADIUS_REMOVAL)
    pcd.visualize(title='Outliers removed - radius', save_as_png=True)

    # Cluster k means the planes
    pcd.estimate_normals(visualize_normals=True, orientate_not_middle=True)
    pcd.cluster_kmeans_normals(remove_ground=True)
    pcd.visualize(title='k-means cluster', original_colors=False)

    # Initialize the ComparePCDMesh class
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    compare.visualize(title='Check pcd and mesh', save_as_png=True)
    compare.check_bricks(points_per_brick=c.POINTS_PER_BRICK)
    compare.visualize_result(filename_vis=c.FILENAME_VIS, save_as_png=True)

    compare.write_results()



if __name__ == '__main__':
    main()