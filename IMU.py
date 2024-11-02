import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Initialiseer pipeline en configureer voor accelerometer en gyroscoop
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.accel)
pipeline.start(config)

# Setup voor live 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

# Initialiseer positie en snelheid
position = np.array([0.0, 0.0, 0.0])
velocity = np.array([0.0, 0.0, 0.0])
orientation = np.array([0.0, 0.0, 0.0])  # Oriëntatie vector
path_x, path_y, path_z = [position[0]], [position[1]], [position[2]]

# Parameters voor tijdsinterval
gyro_scale = 1  # Increased scale factor for more responsive movement

try:
    prev_time = time.time()

    while True:
        # Haal de frames op
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        
        if accel_frame and gyro_frame:
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # Acceleration and gyroscope data
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z]) * gyro_scale

            # Integrate to get orientation change, adjusted by time step
            orientation += gyro * dt
            
            # Apply movement to 3D path based on orientation
            movement = orientation * dt
            path_x.append(movement[0])
            path_y.append(movement[1])
            path_z.append(movement[2])

            # Live plot update
            ax.clear()
            ax.plot(path_x, path_y, path_z, color="blue")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X Orientation")
            ax.set_ylabel("Y Orientation")
            ax.set_zlabel("Z Orientation")
            plt.pause(0.001)  # Faster update rate

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pipeline.stop()
    plt.show()
