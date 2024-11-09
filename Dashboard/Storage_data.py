import matplotlib.pyplot as plt

def plot_storage_data(data):
    '''
    This function visualizes the storage capacity of the robot by creating a horizontal bar chart showing 
    the amount of used and available storage based on the input DataFrame.

    Parameters:
    data (pd.DataFrame): A Pandas DataFrame with the following columns:
                         - 'Total Storage (GB)' (float or integer): Total storage capacity of the robot in GB.
                         - 'Used Storage (GB)' (float or integer): Currently used storage in GB.

    Returns:
    fig (matplotlib.figure.Figure): A matplotlib figure containing a horizontal bar chart representing 
                                    used and available storage with labels and color coding.
    '''

    # Extract storage data from the DataFrame
    total_storage = data['Total Storage (GB)'].iloc[0]
    used_storage = data['Used Storage (GB)'].iloc[0]

    # Calculate available storage
    available_storage = total_storage - used_storage

    # Data for the bar chart (used vs available)
    labels = ['Used Storage', 'Available Storage']
    sizes = [used_storage, available_storage]
    colors = ['grey', '#36607D']

    # Create a horizontal bar chart to represent the storage capacity
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot a horizontal bar
    ax.barh([0], [used_storage], color=colors[0], label=f'Used Storage ({used_storage:.2f} GB)')
    ax.barh([0], [available_storage], left=used_storage, color=colors[1], label=f'Available Storage ({available_storage:.2f} GB)')

    # Add labels and title
    ax.set_title('Robot Storage Capacity', fontsize=14)
    ax.set_xlabel('Storage (GB)', fontsize=12)

    # Add a legend below the chart
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    # Add grid lines and format
    ax.set_xlim(0, total_storage)
    ax.grid(True, axis='x', linestyle='--')

    # Remove y-axis ticks and labels for a clean look
    ax.set_yticks([])

    # Show the plot
    plt.tight_layout()
    
    return fig
