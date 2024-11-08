import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the offset for the masonry wall
wall_left = 1
wall_right = 29
wall_bottom = 0.5
wall_top = 4.5

# Define coordinates for the robot's path based on the masonry wall dimensions
path_x = [wall_left - 1, wall_right + 1, wall_right + 1, wall_left - 1, wall_left - 1]
path_y = [wall_bottom - 1, wall_bottom - 1, wall_top + 1, wall_top + 1, wall_bottom - 1]

# Current position of the robot
current_position = (30, 2.5)

# Example errors: (type of error, (x, y) location on the route)
error_list = [
    ("Obstacle", (10, -0.5)),
    ("Obstacle", (30, 1)),
    ("Obstacle", (20, 5.5)),
    ("Obstacle", (0, 2))
]

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 4))

# Draw the original path with offset (in blue)
ax.plot(path_x, path_y, label='Original Route (Robot Path)', linestyle='-', color='blue', zorder=1)

# Add the current position of the robot as a square
ax.scatter(current_position[0], current_position[1], color='red', marker='s', s=100, label='Current Position', zorder=3)

# Add a rectangle for the masonry wall with an offset outward
wall_rect = patches.Rectangle((wall_left, wall_bottom), wall_right - wall_left, wall_top - wall_bottom,
                              linewidth=2, edgecolor='black', facecolor='lightgray', label='Masonry Wall', zorder=0)
ax.add_patch(wall_rect)

# Adjusted path coordinates
adjusted_path_x = []
adjusted_path_y = []

'''
Function: avoid_obstacle
Purpose: Determines the avoidance path based on the position of an obstacle relative to the masonry wall.
Parameters:
    - x (float)
    - y (float)
Returns:
    - tuple: A tuple containing two lists of coordinates (adjusted_x, adjusted_y) for the avoidance path.
            - adjusted_x (list): The x-coordinates of the adjusted path.
            - adjusted_y (list): The y-coordinates of the adjusted path.
'''
def avoid_obstacle(x, y):
    # Determine the path based on the obstacle's position relative to the defined route
    if y < wall_bottom:  # Obstacle on the bottom horizontal line
        return [x - 1, x - 1, x + 1, x + 1, wall_right + 1], [y, y - 1, y - 1, y, y]
    elif x > wall_right:  # Obstacle on the right vertical line
        return [x, x + 1, x + 1, x, x], [y - 1, y - 1, y + 1, y + 1, wall_top + 1]
    elif y > wall_top:  # Obstacle on the top horizontal line
        return [x + 1, x + 1, x - 1, x - 1, wall_left - 1], [y, y + 1, y + 1, y, y]
    elif x < wall_left:  # Obstacle on the left vertical line
        return [x, x - 1, x - 1, x, x], [y + 1, y + 1, y - 1, y - 1, wall_bottom - 1]

    return [], []  # Return empty if no avoidance is needed

# Plot errors (obstacles) on the route
first_obstacle = True
for error in error_list:
    error_type, error_position = error
    
    if first_obstacle:
        ax.scatter(error_position[0], error_position[1], color='black', marker='x', s=100, label='Obstacle', zorder=2)
        first_obstacle = False
    else:
        ax.scatter(error_position[0], error_position[1], color='black', marker='x', s=100, zorder=2)

    # Avoid the obstacle by adjusting path
    adj_x, adj_y = avoid_obstacle(*error_position)
    
    # Only extend the adjusted path if it has valid coordinates
    if adj_x and adj_y:
        adjusted_path_x.extend(adj_x)
        adjusted_path_y.extend(adj_y)

# Plot the adjusted path to avoid obstacles if there are adjustments made (in green dashed line)
if adjusted_path_x and adjusted_path_y:
    ax.plot(adjusted_path_x, adjusted_path_y, label='Adjusted Route (Avoiding Obstacles)', linestyle='--', color='green', zorder=2)

# Create a text box for displaying the list of errors
error_messages = [f"{error_type} at {pos}" for error_type, pos in error_list]
error_text = "\n".join(error_messages)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# Add the error messages to the plot
ax.text(32, 3, error_text, fontsize=10, bbox=props)

# Labels and title
ax.set_xlabel('X-coordinate (m)')
ax.set_ylabel('Y-coordinate (m)')
ax.set_title('Robot Position and Obstacles with Adjusted Path')

# Add a single legend for all other elements and place it in the upper right corner
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.31, 1.2))

# Add a grid
ax.grid(True)

# Show the plot
plt.show()
