import open3d as o3d
import os
import re

def print_ply_files(output_dir):
    """
    Prints the list of PLY files in the specified directory.

    Args:
        output_dir (str): Path to the directory containing PLY files.
    """
    # Get the list of PLY files in the output directory
    ply_files = [f for f in os.listdir(output_dir) if f.endswith(".ply")]

    if not ply_files:
        print("No PLY files found in the output directory.")
    else:
        # Sort the files numerically based on the scan number, then by filename
        ply_files.sort(key=lambda x: (extract_scan_number(x), x))
        print("PLY files found in the output directory:")
        for file in ply_files:
            print(f"- {file}")

def visualize_pointcloud_from_ply(file_path):
    """
    Visualizes a point cloud from a PLY file using Open3D.

    Args:
        file_path (str): Path to the PLY file.
    """
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud is empty
    if pcd.is_empty():
        print("Point cloud is empty. Please check the file.")
        return

    # Check if colors are present in the point cloud
    if pcd.colors is None or len(pcd.colors) == 0:
        print("No color information found in the point cloud.")
    else:
        print("Visualizing with original colors...")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Point Cloud Visualization",
                                      width=800, height=600)

def extract_scan_number(file_name):
    """
    Extracts the scan number from the file name.

    Args:
        file_name (str): The file name to extract the scan number from.

    Returns:
        int: The scan number or -1 if not found.
    """
    # Match "Scan_" followed by digits (e.g., Scan_1, Scan_10)
    match = re.search(r'Scan_(\d+)', file_name)
    return int(match.group(1)) if match else -1

if __name__ == "__main__":
    # Path to the output directory where PLY files are saved
    output_dir = "./output"

    # Print the list of PLY files in the output directory
    print_ply_files(output_dir)

    # Get the list of PLY files in the output directory
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".ply")]

    if not existing_files:
        print("No PLY files found in the output directory.")
    else:
        # Sort the files numerically based on the scan number
        existing_files.sort(key=extract_scan_number)

        # Visualize the point cloud from the latest PLY file
        latest_file = os.path.join(output_dir, existing_files[-1])  # Get the latest file
        print(f"Visualizing the point cloud from {latest_file}...")
        visualize_pointcloud_from_ply(latest_file)
