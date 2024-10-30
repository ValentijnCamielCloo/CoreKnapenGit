import open3d as o3d
import numpy as np
import os
import datetime
import pyvista as pv
import imageio
# from xx3_functions_registration import draw_registration_result_original_color

# Define the folder paths
scans_folder_path = 'ProgressPilot/ProgressPilot_59_20241030_153617/2_transformed'
output_folder = 'ProgressPilot/ProgressPilot_59_20241030_153617'

# # Create a folder for the current run with date and time
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_folder = os.path.join(output_base_folder, f"ProgressPilotRegistration_{current_time}")
# os.makedirs(output_folder, exist_ok=True)

# Get all PLY files from the transformed scans folder
ply_files = sorted([f for f in os.listdir(scans_folder_path) if f.endswith(".ply")])

# Define multi-scale parameters for Colored ICP registration
voxel_radius = [0.04, 0.02, 0.01]
max_iter = [50, 30, 14]
current_transformation = np.identity(4)

# Initialize cumulative point cloud with the first scan
cumulative_cloud = o3d.io.read_point_cloud(os.path.join(scans_folder_path, ply_files[0]))
cumulative_name = ply_files[0]

# Export the initial cumulative cloud
initial_registered_path = os.path.join(output_folder, "Registered_0.ply")
o3d.io.write_point_cloud(initial_registered_path, cumulative_cloud)

# Load and register each scan iteratively
for i in range(1, len(ply_files)-1):
    source_path = os.path.join(scans_folder_path, ply_files[i])
    source = o3d.io.read_point_cloud(source_path)
    source_name = ply_files[i]
    print(f"Registering {cumulative_name} to {source_name}")

    # Visualize the source cloud before registration
    initial_visualization = f"INITIAL ALIGNMENT: Source: {source_name}, Target: {cumulative_name}"
    # o3d.visualization.draw_geometries([source, cumulative_cloud], window_name=initial_visualization)

    # Multi-scale registration using Colored ICP
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print(f"Scale {scale + 1} - Iterations: {iter}, Voxel size: {radius}")

        # Before registration, show current counts
        print(f"Before registration: Cumulative cloud has {len(cumulative_cloud.points)} points, Source has {len(source.points)} points.")

        # Downsample cumulative cloud and source cloud
        cumulative_down = cumulative_cloud.voxel_down_sample(radius)
        source_down = source.voxel_down_sample(radius)

        # Estimate normals
        cumulative_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Apply Colored ICP registration
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, cumulative_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        print(f"ICP Result for Scale {scale + 1}:")
        print(f"  Transformation matrix:\n{current_transformation}")
        print(f"  Fitness: {result_icp.fitness}, Inlier RMSE: {result_icp.inlier_rmse}")

        # Visualize after each scale
        scale_visualization = f"SCALE {scale + 1}: Source: {source_name}, Target: {cumulative_name} (after scale {scale + 1})"
        transformed_source = source_down.transform(current_transformation)
        # o3d.visualization.draw_geometries([transformed_source, cumulative_down], window_name=scale_visualization)
        # Initialize the plotter in off-screen mode
        # plotter = pv.Plotter(off_screen=True)
        # plotter.clear()  # Clear previous points
        # points_source = np.asarray(transformed_source.points)
        # point_cloud = pv.PolyData(points_source)
        # points_cum = np.asarray(cumulative_down.points)
        # point_cloud_cum = pv.PolyData(points_cum)
        # plotter.add_points(point_cloud)
        # plotter.add_points(point_cloud_cum)

        # # Save the current visualization as an image
        # image_path = f"registration_step_{i+scale}.png"
        # plotter.show(screenshot=image_path, auto_close=False)

    # Transform the source to align with the cumulative point cloud
    source.transform(current_transformation)

    # Combine the cumulative cloud with the registered source
    combined_cloud = cumulative_cloud + source

    # Visualize the registered source and cumulative cloud
    window_title_registration = f"REGISTERED: Source: {source_name}, Target: {cumulative_name} (after registration)"
    # o3d.visualization.draw_geometries([combined_cloud], window_name=window_title_registration)

    # image_folder = r"C:\Users\sarah\PycharmProjects\CoreKnapenGit"
    # # Use imageio.v2.imread to explicitly handle potential version issues
    # images = []
    # # List all .png files in the directory
    # image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    #
    # # Read each image and add to the images list
    # for image in image_paths:
    #     try:
    #         img = imageio.v2.imread(image)
    #         images.append(img)
    #     except Exception as e:
    #         print(f"Error reading {image}: {e}")
    #
    # # Create the GIF if images were successfully read
    # output_gif_path = os.path.join(image_folder, "registration_process.gif")
    # if images:
    #     imageio.mimsave(output_gif_path, images, duration=0.5)
    #     print(f"GIF saved at {output_gif_path}")
    # else:
    #     print("No valid images found to create a GIF.")

    # Export the combined registered source
    registered_output_path = os.path.join(output_folder, f"Registered_{i}.ply")
    # o3d.io.write_point_cloud(registered_output_path, combined_cloud)

    # Update cumulative_cloud and cumulative_name with the new registered source
    cumulative_cloud = combined_cloud
    cumulative_name = f"Registered_{i}.ply"

o3d.visualization.draw_geometries([cumulative_cloud])

print("Registration complete. All registered point clouds are exported.")