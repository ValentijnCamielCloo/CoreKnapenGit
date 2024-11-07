import pandas as pd
import matplotlib.pyplot as plt

def plot_scan_data(df):
    # Calculate total time spent on scans (excluding the last scan)
    if 'Duration' in df.columns and df['Duration'].notnull().any():
        total_time = df['Duration'].iloc[:-1].sum()
    else:
        total_time = 0

    # Current scan duration (the last valid entry)
    current_scan_duration_index = df['Duration'].last_valid_index()
    current_scan_duration = df['Duration'].iloc[current_scan_duration_index] if current_scan_duration_index is not None else 0

    # Average duration of all scans (excluding the last scan)
    average_duration = df['Duration'].mean() if 'Duration' in df.columns and df['Duration'].notnull().any() else 0

    # Remaining scans to be made
    remaining_scans = df.shape[0] - 1
    if current_scan_duration_index is not None:
        remaining_scans -= 1

    # Total time to complete remaining scans
    total_time_remaining = remaining_scans * average_duration if remaining_scans > 0 and average_duration > 0 else 0

    # Create a display for the results
    fig, ax = plt.subplots(figsize=(6, 4))

    # Display the information with a blank line added
    display_text = (f"Total Time for Previous Scans: {total_time:.2f} seconds\n"
                     f"Current Scan Duration: {current_scan_duration:.2f} seconds\n"
                     f"Average Duration: {average_duration:.2f} seconds\n\n"  
                     f"Remaining Scans: {remaining_scans}\n"
                     f"Estimated Total Time for Remaining Scans: {total_time_remaining:.2f} seconds")

    ax.text(0.5, 0.5, display_text, fontsize=12, ha='center', va='center', color='#36607D')
    ax.set_axis_off()

    return fig