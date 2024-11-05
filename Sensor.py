# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import constants as c
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

import os
from datetime import datetime

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)



def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    # Calculate distances from the camera (assuming camera is at origin [0,0,0])
    distances = np.linalg.norm(verts, axis=1)
    
    # Create a mask to filter out points at a distance of 2 meters or more THIS ONLY WORKS IN THE VISUALIZER NOT THE EXPORT
    mask = distances < c.FILTERED_DISTANCE  # Keep points that are less than x meters

    # Apply the mask to the vertices and texture coordinates
    verts = verts[mask]
    texcoords = texcoords[mask]

    if painter:
        # Painter's algo, sort points from back to front
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]

    # proj now contains 2D image coordinates
    j, i = proj.astype(np.uint32).T

    # Create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # Sort texcoord with same indices as above
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T

    # Clip texcoords to image
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    # Perform UV-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


# Call this function with appropriate parameters

out = np.empty((h, w, 3), dtype=np.uint8)

#ADDED 

output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#ADDED 

def save_filtered_ply(filename, vertices, texcoords, colors=None):
    with open(filename, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # Check if colors are provided and write accordingly
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")

        # Transform coordinates for Cartesian system (swap y and z)
        for i in range(len(vertices)):
            x, y, z = vertices[i]
            # Apply transformation: y (depth) -> z, z -> y
            f.write("{} {} {}\n".format(x, z, -y))  # Invert y for correct orientation
            # Write color data if provided
            if colors is not None:
                r, g, b = colors[i]
                f.write("{} {} {}\n".format(int(r), int(g), int(b)))
#ADDED


while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        
        #ADDED 
        color_data = np.asanyarray(color_frame.get_data()).reshape(-1, 3)
        # Calculate distances and filter points
        distances = np.linalg.norm(verts, axis=1)
        mask = distances < c.FILTERED_DISTANCE  # Keep points that are less than x meters
        
        # Convert texcoords to pixel positions in the color image
        cw, ch = color_frame.get_width(), color_frame.get_height()  # Assuming 1920x1080
        u, v = texcoords[:, 0], texcoords[:, 1]
        u = np.clip(u * cw, 0, cw - 1).astype(np.int32)
        v = np.clip(v * ch, 0, ch - 1).astype(np.int32)

        # Use texcoords to map verts to corresponding colors from the color frame
        colors = color_data[v * cw + u]  # This maps the 3D points to their color values


        #DEBUGGING
        # Before filtering vertices and texture coordinates, let's add debug statements
        print(f"Shape of verts: {verts.shape}")
        print(f"Shape of texcoords: {texcoords.shape}")
        print(f"Shape of color_data: {color_data.shape}")
        print(f"Length of mask: {len(mask)}, Sum of mask (filtered points): {np.sum(mask)}")
        #DEBUGGING

        # Apply mask to verts and texcoords as before
        
        verts_filtered = verts[mask]
        texcoords_filtered = texcoords[mask]
        colors_filtered = colors[mask]
        #colors_filtered = color_data[:len(verts)][mask]

        # Here is where the issue is likely happening.
        # The shape of color_data must match the number of verts.
        # We need to debug this by comparing shapes directly.
        # Uncomment the following debug line:

        print(f"Mask: {mask}")
        print(f"Filtered Verts Shape: {verts_filtered.shape}")
        print(f"Filtered texcoords shape: {texcoords_filtered.shape}")
        print(f"Filtered colors filtered shape: {colors_filtered.shape}")

        # Commented problematic line - colors_filtered seems to not match the shape correctly:
        # The correct approach depends on what color_data represents (RGB values).
        # Let's first ensure color_data is properly aligned to verts, then we can filter it as needed.
        #ADDED 


    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)



    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        # Get existing filtered PLY files
        filtered_files = [f for f in os.listdir(output_dir) if f.endswith("_filtered.ply")]

        # Extract scan numbers and find the maximum
        scan_numbers = []
        for f in filtered_files:
            parts = f.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                scan_numbers.append(int(parts[1]))

        # Determine the next scan number
        if scan_numbers:
            scan_number = max(scan_numbers) + 1
        else:
            scan_number = 1  # Start from 1 if no existing files

        # Get the current date and time in the desired format
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Format the filename with scan number and date for original point cloud
        original_filename = f"Scan_{scan_number}_{current_date}_original.ply"
        full_original_path = os.path.join(output_dir, original_filename)

        # Export the original point cloud data
        print(f"Exporting original {original_filename} .... ")
        points.export_to_ply(full_original_path, mapped_frame)

        # Format the filename for filtered data
        filtered_filename = f"Scan_{scan_number}_{current_date}_filtered.ply"
        full_filtered_path = os.path.join(output_dir, filtered_filename)

        # Save the filtered vertices and texture coordinates
        print(f"Exporting filtered {filtered_filename} .... ")
        # Save the filtered vertices and texture coordinates
        # save_filtered_ply(full_filtered_path, verts_filtered, texcoords_filtered, colors_filtered)
        save_filtered_ply(full_filtered_path, verts_filtered, texcoords_filtered, colors_filtered)


    
    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()