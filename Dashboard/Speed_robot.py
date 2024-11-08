# Speed_robot.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_robot_speed(df):
    '''
    This function creates a plot displaying the robot's speed over time. It extracts the 'Time' and 'Speed' columns 
    from the provided DataFrame and uses them to plot the speed as a line with markers. Additionally, it displays bars 
    below each point to indicate the speed at each time interval.

    Parameters:
    df (pd.DataFrame): A Pandas DataFrame containing at least 'Time' and 'Speed' columns. 
                       'Time' should represent time intervals (in seconds) and 'Speed' the robot's speed 
                       at those times (in meters per second). Both columns should be numeric (integer or float).

    Returns:
    fig (matplotlib.figure.Figure): A matplotlib figure showing the speed of the robot over time, with labeled axes, 
                                    a title, a grid, and a legend.
    '''

    # Extract the 'Time' and 'Speed' columns
    time = df['Time']
    speed = df['Speed']
    
    # Create the plot for speed over time
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the speed with markers
    ax.plot(time, speed, label='Robot Speed', color='#36607D', marker='o')
    
    # Add bars below each point
    bar_width = 0.3
    for i, s in enumerate(speed):
        ax.bar(time[i], s, width=bar_width, color='#36607D', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Robot Speed Over Time')
    
    # Add grid and legend
    ax.grid(True)
    ax.legend()
    
    return fig
