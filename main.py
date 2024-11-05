from functions import *
import constants as c
from gif import gif

def main():
    # Create an instance of the point cloud
    # pcd = PointCloud()
    # pcd.load_pcd(scan_dir=c.SCAN_DIR)
    # pcd.visualize(title='scans', save_as_png=True)
    # gif('scans', title='Scan')

    # gif('colorized', title='Colorized scan')

    # # # Correct the colors in the point clouds
    # # pcd.colorize()
    # # pcd.visualize(title='Colorized pcd', save_as_png=True)
    #
    # Create an instance of the point cloud
    pcd = PointCloud()
    pcd.load_pcd(scan_dir=c.COLORIZED_DIR)
    # pcd.visualize(title='Colorized loaded pcd', save_as_png=True)

    # Translate and orientate the point cloud to the same place as the model
    pcd.translate_orientate()
    pcd.visualize(title='Translated and orientated pcd', save_as_png=True)

    # Remove ground (black points)
    pcd.filter_colors(filter_color=c.FILTER_COLOR, color_threshold=c.COLOR_THRESHOLD)
    pcd.visualize(title='Filtered on colors', save_as_png=True)

    # Registration of the scans to create one point cloud
    pcd.registration()
    pcd.visualize(title='Registered', save_as_png=True)

    # Downsample and filter point cloud
    pcd.voxel_downsample(voxel_size=c.VOXEL_SIZE)
    pcd.visualize(title='Downsampled', save_as_png=True, rotate=True)

    # Cluster the four sides
    pcd.estimate_normals(orientate_not_middle=True, visualize_normals=False)
    pcd.cluster_kmeans_normals(show_elbow=True)
    pcd.visualize(title='K-means clustered', save_as_png=True, original_colors=False, rotate=True)

    # Cluster separate clusters to filter out outliers
    pcd.cluster_kmeans_normals(biggest_cluster=True)
    pcd.visualize(title='K-means outlier removal', save_as_png=True, original_colors=False, rotate=True)

    # Translate the point cloud to origin
    pcd.translate_to_origin(dist_scanner_obj=c.DIST_SCANNER_OBJ,height_scanner=c.HEIGHT_SCANNER)

    # Create an instance of the Mesh
    meshes = Mesh()
    meshes.load_meshes()

    # Create an instance for comparing mesh and pcd
    compare = ComparePCDMesh(pcd.pcd, meshes.meshes)
    compare.visualize(title='Check translated pcd', save_as_png=True, original_colors=False, rotate=True)

    compare.check_bricks(points_per_brick=c.POINTS_PER_BRICK)
    compare.visualize_result(filename_vis=c.FILENAME_VIS, save_as_png=True, rotate=True)

    # Write results to a csv
    compare.write_results()



if __name__ == '__main__':
    main()