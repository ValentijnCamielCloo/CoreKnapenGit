# Point Cloud processing
FILE_NAME_PCD = 'Scan_2_20241017_122833_filtered.ply'

# Voxel downsampling
VOXEL_SIZE = 0.005

# Radius outlier removal
NB_POINTS = 15
RADIUS_RADIUS_REMOVAL = 0.015

# Normal outlier removal
RADIUS_NORMAL_REMOVAL = 0.2
THRESHOLD_ANGLE = 10.0
MAX_NN = 30

# DBSCAN clustering
EPS = 0.02
MIN_SAMPLES = 15
MIN_POINTS = 50

# Comparing Mesh and Point Cloud
POINTS_PER_BRICK = 15
FILE_NAME_VIS = '26102024_vis_model.ply'