from functions import *
import constants as c

def main():
    # Create an instance of the point cloud
    pcd = PointCloud()
    pcd.load_pcd(filename=c.FILENAME_PCD)

    # Downsample and filter point cloud
    pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    pcd.visualize(title='Downsampled', save_as_png=True)

    # Filter on color
    pcd.filter_colors(filter_color=c.FILTER_COLOR,color_threshold=c.COLOR_THRESHOLD)
    pcd.visualize(title='Filtered on colors', save_as_png=True)

    # Cluster the four sides
    pcd.estimate_normals(orientate_not_middle=True, visualize_normals=False)
    pcd.cluster_kmeans_normals(show_elbow=True)
    pcd.visualize(title='K-means clustered', save_as_png=True, original_colors=False)

    # Cluster separate clusters to filter out outliers
    pcd.cluster_kmeans_normals(biggest_cluster=True)
    pcd.visualize(title='K-means outlier removal', save_as_png=True, original_colors=False)

    # Translate the point cloud to origin
    pcd.translate(dist_scanner_obj=c.DIST_SCANNER_OBJ,height_scanner=c.HEIGHT_SCANNER)

    # Create an instance of the Mesh
    meshes = Mesh()
    meshes.load_meshes()

    # Create an instance for comparing mesh and pcd
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    compare.visualize(title='Check translated pcd', save_as_png=True, original_colors=False)

    compare.check_bricks(points_per_brick=c.POINTS_PER_BRICK)
    compare.visualize_result(filename_vis=c.FILENAME_VIS, save_as_png=True)

    # Write results to a csv
    compare.write_results()



if __name__ == '__main__':
    main()