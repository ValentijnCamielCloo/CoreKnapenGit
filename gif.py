import os
import imageio
import numpy as np
import pyvista as pv
import open3d as o3d
import cv2

def gif(scan_dir, title):
    # Define the output directory for images and create it if it doesn't exist
    output_dir = 'gif_test'
    os.makedirs(output_dir, exist_ok=True)

    # List all .ply files in the scans directory
    scan_dir = scan_dir  # Directory containing your .ply scan files
    scan_files = sorted([f for f in os.listdir(scan_dir) if f.endswith('.ply')])

    # Loop through each scan file to create and save images
    for i, scan in enumerate(scan_files):
        file_path = os.path.join(scan_dir, scan)  # Corrected path to read point cloud from scans directory
        pc = o3d.io.read_point_cloud(file_path)  # Load point cloud

        # Initialize the plotter in off-screen mode
        plotter = pv.Plotter(off_screen=True)
        plotter.clear()  # Clear previous points
        points_source = np.asarray(pc.points)
        point_cloud = pv.PolyData(points_source)

        # Check and apply colors if available in the point cloud
        if pc.has_colors():
            colors = np.asarray(pc.colors)  # Colors in Open3D are normalized (0-1)
            colors = (colors * 255).astype(np.uint8)  # Convert to 0-255 for PyVista
            if colors is not None:
                point_cloud['RGB'] = colors  # Add color data to PyVista object
            plotter.add_points(point_cloud, scalars='RGB', rgb=True)  # Plot with RGB colors
        else:
            plotter.add_points(point_cloud)  # Add points without color if not available

        plotter.add_title(f'{title} {i+1}', font_size=12)

        # Set the camera position slightly above the origin for this frame
        plotter.camera.position = (0, 0, 0.03)  # Adjusted position to be slightly higher (1 unit above the origin)
        plotter.camera.focal_point = np.mean(points_source, axis=0)  # Focus on the center of the point cloud
        plotter.camera.view_up = (0, 0, 1)  # Define the up direction along the Z-axis

        # Update the plotter to ensure everything is set correctly
        plotter.reset_camera()

        plotter.zoom_camera(1.3)

        # Save the current visualization as an image in the output directory
        image_path = os.path.join(output_dir, f"{scan_dir}_{i}.png")
        plotter.show(screenshot=image_path, auto_close=True)  # Set auto_close=True to close plotter after saving

    # Create a GIF from saved images
    images = []
    image_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(f'{scan_dir}_')])

    # Read each image and add it to the images list
    for image_path in image_paths:
        try:
            img = imageio.v2.imread(image_path)  # Use imageio.v2 to handle potential version issues
            images.extend([img] * 20)  # Duplicate each frame 20 times to slow down playback
        except Exception as e:
            print(f"Error reading {image_path}: {e}")

    # Save the GIF with a long duration per frame to slow down playback
    filename = scan_dir.replace(" ", "_") + ".png"
    output_gif_path = os.path.join(output_dir, f"{filename}.gif")
    if images:
        imageio.mimsave(output_gif_path, images, duration=2)  # 2 seconds per frame
        print(f"GIF saved at {output_gif_path}")
    else:
        print("No valid images found to create a GIF.")

    # Optional: Save as MP4 video for consistent playback speed
    output_video_path = os.path.join(output_dir, f"{filename}.mp4")
    if images:
        imageio.mimsave(output_video_path, images, fps=3)  # 3 frames per second
        print(f"Video saved at {output_video_path}")

    # Open and play the video
    cap = cv2.VideoCapture(output_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
