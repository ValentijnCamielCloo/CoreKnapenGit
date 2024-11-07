import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_battery_percentage(df):
    # Ensure the columns 'Time' and 'Battery Percentage' are present in the Excel file
    iterations = df['Time']
    battery_levels = df['Battery Percentage']

    # Get the battery percentage from the first row of the appropriate column (index 0)
    battery_percentage = df['Battery level'].iloc[0]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the battery level line on the first subplot
    axs[0].plot(iterations, battery_levels, label='Battery Percentage', color='#36607D', marker='o')

    # Add bars under each point
    bar_width = 0.5
    for i, level in enumerate(battery_levels):
        axs[0].bar(iterations[i], level, width=bar_width, color='#36607D', alpha=0.3)

    # Labels and title for the first subplot
    axs[0].set_xlabel('Time (min)')
    axs[0].set_ylabel('Battery Percentage')
    axs[0].set_title('Battery Level of the Robot')
    axs[0].grid(True)
    axs[0].legend()

    # Function to draw a vertical battery
    def draw_vertical_battery(ax, percentage):
        # Ensure that percentage is between 0 and 100
        percentage = max(0, min(percentage, 100))

        # Define battery parameters
        battery_width = 2
        battery_height = 4
        terminal_width = 0.8
        terminal_height = 0.2
        offset = 0.1

        # Draw the battery container
        battery_container = patches.Rectangle((0, 0), battery_width, battery_height, fill=False, linewidth=2, edgecolor='#36607D')
        ax.add_patch(battery_container)

        # Draw the battery terminal above the battery
        terminal = patches.Rectangle((0.6, battery_height), terminal_width, terminal_height, fill=True, color='#36607D')
        ax.add_patch(terminal)

        # Calculate the height of the battery fill based on the percentage with offset
        fill_height = (percentage / 100) * (battery_height - 2 * offset)
        fill_y_position = offset

        # Draw the battery fill
        battery_fill_color = '#36607D'
        battery_fill = patches.Rectangle((0.1, fill_y_position), battery_width - 0.2, fill_height, fill=True, color=battery_fill_color if percentage > 20 else 'red')
        ax.add_patch(battery_fill)

        # Add percentage text
        ax.text(battery_width / 2, battery_height / 2, f'{percentage}%', ha='center', va='center', fontsize=15)

        # Set the axes
        ax.set_xlim(-1, battery_width + 1)
        ax.set_ylim(-1, battery_height + 1)
        ax.set_aspect('equal')
        ax.axis('off')

    # Draw the battery with the battery percentage on the second subplot
    draw_vertical_battery(axs[1], battery_percentage)

    # Set the title for the second subplot
    axs[1].set_title('Battery Level')

    # Adjust the layout to reduce the space between subplots
    plt.subplots_adjust(wspace=0.05)  # Reduce wspace for closer subplots

    return fig  # Return the figure for plotting in Streamlit
