# Point Cloud processing
FILE_DIR_PCD = r'D:\TUdelftGitCore\CoreKnapenGit\scans'
FILE_NAME_PCD = 'Scan_7_20241017_123604_filtered.ply'

# Sensor depth mask
FILTERED_DISTANCE = 0.8

# Voxel downsampling
VOXEL_SIZE = 0.01

# Radius outlier removal CORNERS)
NB_POINTS = 15
RADIUS = 0.02

# Radius outlier removal CORNERS)
NB_POINTS_SP = 15
RADIUS_SP = 0.025

# Registration threshold ICP
THRESHOLD_ICP = 0.1

# DBSCAN clustering
EPS = 0.02
MIN_SAMPLES = 15
MIN_POINTS = 70

# Mesh processing
FILE_DIR_MESH = r'D:\TUdelftGitCore\CoreKnapenGit\model'
FILE_NAME_MESH_LIST = ['composite_meshes2-1_15-10.ply', 'composite_meshes2-2_15-10.ply']

# Comparing Mesh and Point Cloud