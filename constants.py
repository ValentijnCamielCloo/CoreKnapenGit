# Point Cloud processing
FILENAME_PCD = "Registered_4_aligned_to_Scan_2.ply"

DIST_SCANNER_OBJ = 0.3
HEIGHT_SCANNER = 0.045

# Voxel downsampling
VOXEL_SIZE = 0.005

# Filter based on colors
FILTER_COLOR = (0,0,0)  # black
COLOR_THRESHOLD = 0.1

# Radius outlier removal
NB_POINTS = 25
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
FILENAME_VIS = '26102024_vis_model.ply'