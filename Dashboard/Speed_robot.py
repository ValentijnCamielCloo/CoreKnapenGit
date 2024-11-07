# Speed_robot.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_robot_speed(df):
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