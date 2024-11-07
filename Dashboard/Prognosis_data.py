import pandas as pd
import matplotlib.pyplot as plt

def plot_prognosis(data):
    # Ensure that data is a DataFrame
    if isinstance(data, str):
        try:
            data = pd.read_csv(data, sep=';')
        except FileNotFoundError:
            print(f"File not found: {data}. Please check the file path and name.")
            raise
        except pd.errors.ParserError:
            print(f"Error parsing the CSV file: {data}. Please check the format.")
            raise
    
    print("Loaded columns:", data.columns)

    # Ensure that the required columns exist
    required_columns = ['Wall', 'Percentage Completed', 'Total Percentage', 'Bricks Built', 'Bricks to be Built']
    if not all(column in data.columns for column in required_columns):
        print(f"Error: One or more required columns {required_columns} are missing in the 'Prognosis' data.")
        raise KeyError("Missing required columns")

    # Number of walls to plot
    walls = data['Wall'].unique()

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(len(walls), 2, figsize=(10, len(walls) * 4))

    for i, wall in enumerate(walls):
        # Filter data for each wall
        wall_data = data[data['Wall'] == wall].iloc[0]

        # Extract relevant data
        completed_percentage = wall_data['Percentage Completed']
        total_percentage = wall_data['Total Percentage']
        remaining_percentage = total_percentage - completed_percentage

        # Bar chart (progress)
        ax_bar = axes[i, 0]
        ax_bar.barh([0], [completed_percentage], color='#36607D', label=f'Completed ({completed_percentage:.2f}%)')
        ax_bar.barh([0], [remaining_percentage], left=completed_percentage, color='lightgray', label=f'Remaining ({remaining_percentage:.2f}%)')

        # Customize bar chart
        ax_bar.set_title(f'Progress for {wall}', fontsize=14)
        ax_bar.set_xlabel('Percentage (%)', fontsize=12)
        ax_bar.set_xlim(0, total_percentage)
        ax_bar.grid(True, axis='x', linestyle='--')
        ax_bar.set_yticks([])
        ax_bar.legend(loc='center', bbox_to_anchor=(0.5, -0.3), ncol=2)

        # Text display (bricks) - Right column
        bricks_built = wall_data['Bricks Built']
        bricks_to_be_built = wall_data['Bricks to be Built']

        # Add the text
        ax_text = axes[i, 1]
        ax_text.text(0.5, 0.5, f'{bricks_built} / {bricks_to_be_built}', 
                     horizontalalignment='center', verticalalignment='center', fontsize=24, 
                     fontweight='bold', color='#36607D')

        # Add title for the text display
        ax_text.set_title(f'Bricks Laid for {wall}', fontsize=14)

        # Remove the axis lines and ticks for a clean text display
        ax_text.set_axis_off()

    # Adjust layout
    plt.tight_layout()
    return fig
