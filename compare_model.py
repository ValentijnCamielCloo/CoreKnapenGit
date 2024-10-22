import open3d as o3d
import numpy as np
import pyvista as pv
import os

def compare_model(input_name_model, input_name_scan, input_name_model_vis):
    # Define the folder path where your PLY files are stored and where the filtered PLY will be saved
    input_folder_model = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\comparing_model'
    input_folder_scan = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\saved_meshes'
    # output_folder = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\saved_meshes'

    input_folder_model_vis = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\visualizing_model'
    # input_folder_scan_vis = r'C:\Users\sarah\PycharmProjects\CoreKnapenGit\_point_clouds'
    # os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Specify the mesh file name to load
    file_path_model = os.path.join(input_folder_model, f"{input_name_model}.ply")
    file_path_scan = os.path.join(input_folder_scan, f"{input_name_scan}.ply")

    file_path_model_vis = os.path.join(input_folder_model_vis, f"{input_name_model_vis}.ply")
    # file_path_scan_vis = os.path.join(input_folder_scan_vis, f"{input_name_scan_vis}.ply")
    # Create a plotter object for visualization
    plotter = pv.Plotter()

    # Load the meshes
    mesh_model = pv.read(file_path_model)
    mesh_scan = pv.read(file_path_scan)

    mesh_model_vis = pv.read(file_path_model_vis)
    # mesh_scan_vis = pv.read(file_path_scan_vis)

    area_model = round(mesh_model.area,2)
    print(f"Total mesh area model: {area_model} m2")
    area_scan = round(mesh_scan.area,2)
    print(f"Total mesh area scan: {area_scan} m2")
    area_to_built = area_model - area_scan

    # Calculate the percentage built
    percentage_built = (area_scan / area_model) * 100
    print(f"The progress is: {round(percentage_built, 2)} %")

    # Add the model mesh (yellow)
    plotter.add_mesh(mesh_model_vis, color='yellow', opacity=0.9)

    # Add the scan mesh (red)
    plotter.add_mesh(mesh_scan)

    # Add progress
    plotter.add_text(f"Progress: {percentage_built:.2f}% \n {area_to_built} m2 still needs to be built", position='upper_right', font_size=10, color='black')

    # Set background and add additional information
    plotter.set_background("white")
    plotter.add_text(f"Yellow: Model, area: {round(area_model,2)} m2", position='lower_left', color='black', font_size=10)
    plotter.add_text(f"Red: Scan, area: {round(area_scan,2)} m2", position='lower_right', color='black', font_size=10)

    # Display the plot
    plotter.show()

compare_model('model1_08-10', 'mesh_from_pcd1_08-10', 'visualizing_model1_08-10')