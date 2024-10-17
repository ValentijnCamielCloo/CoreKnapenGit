from filter_pcd import *
from translate_pcd import *
from pcd_to_mesh import *
from compare_model import *

# Filter the point cloud, only keep point within certain distance from scanner
# For further filtering we use voxel downsampling and statistical outlier removal
filter_dist_pcd = filter_pcd('Scan_31_20241011_105409_filtered', 'filtered_pcd1_11-10', 0.37, 0.005)

# Move all the points to one plane and let the corner (left or right) start in the origin (0,0)
translate_pcd('filtered_pcd1_11-10','translated_pcd1_11-10','Right')

# Create a mesh from the point cloud
pcd_to_mesh('translated_pcd1_11-10', 'mesh_from_pcd1_11-10')

# Compare the mesh from the scan to the final model
# Output: percentage built and m2 to be built
compare_model('composite_meshes_right_11-10', 'mesh_from_pcd1_11-10', 'visualizing_model1_08-10')