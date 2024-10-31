import open3d as o3d
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the folder paths
scans_folder_path = r"D:\TUdelftGitCore\CoreKnapenGit\scans"
output_folder_path = r"D:\TUdelftGitCore\CoreKnapenGit\Colored"

# Ensure output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Get all .ply files in the directory
ply_files = glob.glob(os.path.join(scans_folder_path, "*.ply"))

# Load scale values from CSV
scale_df = pd.read_csv(r"D:\TUdelftGitCore\CoreKnapenGit\Scan_Color_scale.csv")
scale_values = {row['Scan']: float(row['Scale']) for index, row in scale_df.iterrows()}  # Ensure scale is float

# Iterate through all PLY files
for ply_file in ply_files:
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(f"Processing: {ply_file}")

    # Get the colors from the point cloud
    colors = np.asarray(point_cloud.colors)

    # Get the scan name from the file name
    scan_name = "_".join(os.path.basename(ply_file).split('_')[0:2])  # e.g., "Scan_4"

    # Get the corresponding scale value
    scale_value = scale_values.get(scan_name, 4.1)  # Default scale if not found

    # Debug statement for scale value
    if scan_name in scale_values:
        print(f"Found scale value for {scan_name}: {scale_value}")
    else:
        print(f"No scale value found for {scan_name}. Using default scale value: {scale_value}")

    # Scale the colors
    colors_scaled = np.clip(colors * scale_value, 0, 1)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_scaled)

    # Filter out white colors
    non_white_mask = ~np.all(colors_scaled > 0.95, axis=1)
    non_white_colors = colors_scaled[non_white_mask]

    # Store red channel averages
    red_channel_values = []

    # Number of runs
    num_runs = 50
    for run in range(num_runs):
        # K-means clustering
        num_colors = 4
        kmeans = KMeans(n_clusters=num_colors, random_state=run)
        kmeans.fit(non_white_colors)

        color_labels = kmeans.labels_
        main_colors = kmeans.cluster_centers_

        # Collect red channel averages
        for i in range(num_colors):
            avg_rgb = (main_colors[i] * 255).astype(int)
            red_channel = avg_rgb[0]
            red_channel_values.append(red_channel)

    # Estimate thresholds based on quantiles
    red_channel_values = np.array(red_channel_values)
    q1, q2, q3 = np.percentile(red_channel_values, [25, 50, 75])
    ranges = [
        (0, q1),
        (q1, q2),
        (q2, q3),
        (q3, 255)
    ]

    # Prepare lists to store red channel values for each assigned color
    color_hist_data = {
        'Red': [],
        'Dark Green': [],
        'Yellow': [],
        'Blue': [],
    }

    # Function to assign color based on estimated ranges
    def assign_color_based_on_red(red_value):
        if ranges[0][0] <= red_value < ranges[0][1]:
            color_hist_data['Red'].append(red_value)
            return (255, 0, 0)  # Assign red
        elif ranges[1][0] <= red_value < ranges[1][1]:
            color_hist_data['Dark Green'].append(red_value)
            return (255, 255, 68)  # Assign dark green
        elif ranges[2][0] <= red_value < ranges[2][1]:
            color_hist_data['Yellow'].append(red_value)
            return (35, 157, 64)  # Assign yellow
        else:
            color_hist_data['Blue'].append(red_value)
            return (0, 0, 255)  # Default to blue if no match

    # Final coloring process based on average red channel values
    uniform_colors = np.zeros_like(colors)
    for run in range(num_runs):
        # K-means clustering again for coloring
        kmeans = KMeans(n_clusters=num_colors, random_state=run)
        kmeans.fit(non_white_colors)

        color_labels = kmeans.labels_
        for i in range(num_colors):
            avg_rgb = (kmeans.cluster_centers_[i] * 255).astype(int)
            red_channel = avg_rgb[0]
            cluster_points = (color_labels == i)
            original_indices = np.where(non_white_mask)[0][cluster_points]
            assigned_color = assign_color_based_on_red(red_channel)
            
            uniform_colors[original_indices] = np.array(assigned_color) / 255.0  # Normalize for Open3D

    # Apply the colored array to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(uniform_colors)

    # Save the new colored point cloud
    output_file_name = os.path.splitext(os.path.basename(ply_file))[0] + "_colored.ply"
    output_file_path = os.path.join(output_folder_path, output_file_name)
    o3d.io.write_point_cloud(output_file_path, point_cloud)
    print(f"Saved colored point cloud to: {output_file_path}")

    # Visualize the final colored point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    # Set camera parameters for isometric view
    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = np.array([[0.577, -0.577, 0.577, 0],
                                         [0.577, 0.577, -0.577, -3],  # Lower the camera and angle up
                                         [-0.577, -0.577, -0.577, 6],  # Adjust Z for upward facing
                                         [0, 0, 0, 1]])
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    # Capture and save the PNG image with the scale value in the filename
    png_file_path = os.path.join(output_folder_path, f"{scale_value:.1f}_colored.png")  # Save with scale value
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(png_file_path)
    print(f"Saved visualization to: {png_file_path}")
    
    vis.run()
    vis.destroy_window()

# Plot histograms for assigned colors with enhancements
plt.figure(figsize=(12, 8))

# Increase the number of bins for better visibility
num_bins = 30

for color_name, values in color_hist_data.items():
    # Plot each histogram with increased opacity and distinct colors
    plt.hist(values, bins=num_bins, alpha=0.7, label=color_name)

plt.title('Distribution of Red Channel Values for Assigned Colors')
plt.xlabel('Red Channel Value')
plt.ylabel('Frequency')
plt.xlim(0, 255)  # Set x-axis limits to the full range of red channel values
plt.ylim(0, max(len(values) for values in color_hist_data.values()) * 1.1)  # Adjust y-axis limit for clarity
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.show()
