import matplotlib.pyplot as plt

def plot_robot_status(df_status):
    '''
    This function visualizes the current status of the robot, including its speed, connection status, 
    and operational status. It uses the last entries from the 'Speed', 'Status', and 'Connected' columns 
    of the provided DataFrame to generate a status display.

    Parameters:
    df_status (pd.DataFrame): A Pandas DataFrame containing the columns:
                              - 'Speed' (float or integer): The robot's current speed in meters per second.
                              - 'Status' (string): The operational status of the robot (e.g., 'working', 'paused', 'completed', 'error').
                              - 'Connected' (boolean or integer): The connection status of the scanner to the robot (True if connected, False if not).

    Returns:
    fig (matplotlib.figure.Figure): A matplotlib figure displaying the robot's current speed, connection status, 
                                    and operational status as text in a clean, centered layout.
    '''

    # Extract the current speed, robot status, and connection status from the last row of the DataFrame
    current_speed = df_status['Speed'].iloc[-1] if 'Speed' in df_status.columns else 0
    robot_status = df_status['Status'].iloc[-1].lower() if 'Status' in df_status.columns else 'unknown'
    connection_status = df_status['Connected'].iloc[-1] if 'Connected' in df_status.columns else False

    # Create figure and axis for the visualization
    fig, ax = plt.subplots()

    # Display the current speed
    status_text = f"Current Speed of the Robot: {current_speed} m/s"
    ax.text(0.5, 0.7, status_text, fontsize=16, ha='center', va='center', color='#36607D')

    # Display the connection status
    if connection_status:
        connection_message = "Scanner is connected to the robot"
        connection_color = 'green'
    else:
        connection_message = "Scanner is not connected to the robot"
        connection_color = 'red'
        robot_status = 'error'

    ax.text(0.5, 0.5, connection_message, fontsize=14, ha='center', va='center', color=connection_color)

    # Set status message and color based on the robot's operational status
    if robot_status == 'working':
        status_color = 'green'
        status_message = "Status: Working"
    elif robot_status == 'paused':
        status_color = 'yellow'
        status_message = "Status: Paused"
    elif robot_status == 'completed':
        status_color = 'blue'
        status_message = "Status: Completed"
    elif robot_status == 'error':
        status_color = 'red'
        status_message = "Status: Error"
    else:
        status_color = 'gray'
        status_message = "Status: Unknown"

    # Add the status text
    ax.text(0.5, 0.3, status_message, fontsize=14, ha='center', va='center', color=status_color)

    # Remove axes for a clean display
    ax.set_axis_off()
    return fig
