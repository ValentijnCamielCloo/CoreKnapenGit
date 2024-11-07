import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

def plot_planning(file, current_date):  # Added current_date as a second parameter
    current_date = pd.to_datetime(current_date, dayfirst=True)
    
    df_planning = pd.read_csv(file, delimiter=';')

    df_planning['Start Date'] = pd.to_datetime(df_planning['Start Date'], format='%d-%m-%Y', dayfirst=True)
    df_planning['End Date'] = pd.to_datetime(df_planning['End Date'], format='%d-%m-%Y', dayfirst=True)
    df_planning['Current Progress (%)'] = pd.to_numeric(df_planning['Current Progress (%)'], errors='coerce').fillna(0)
    df_planning['Total Duration'] = (df_planning['End Date'] - df_planning['Start Date']).dt.days
    df_planning['Current Progress Duration'] = (df_planning['Current Progress (%)'] / 100) * df_planning['Total Duration']
    df_planning['Remaining Duration'] = df_planning['Total Duration'] - df_planning['Current Progress Duration']

    # Calculate planned progress (%)
    df_planning['Planned Progress (%)'] = 0
    for index, row in df_planning.iterrows():
        if current_date < row['Start Date']:
            df_planning.at[index, 'Planned Progress (%)'] = 0
        elif current_date > row['End Date']:
            df_planning.at[index, 'Planned Progress (%)'] = 100
        else:
            elapsed_days = (current_date - row['Start Date']).days
            total_days = (row['End Date'] - row['Start Date']).days
            progress = (elapsed_days / total_days) * 100 if total_days > 0 else 0
            df_planning.at[index, 'Planned Progress (%)'] = min(progress, 100)

    # Add planned progress duration and deviation
    df_planning['Planned Progress Duration'] = (df_planning['Planned Progress (%)'] / 100) * df_planning['Total Duration']
    df_planning['Deviation'] = df_planning['Current Progress Duration'] - df_planning['Planned Progress Duration']

    # Adjust start dates for tasks following non-overlapping tasks
    masonry_subtasks = ['Masonry Wall 1', 'Masonry Wall 2', 'Masonry Wall 3', 'Masonry Wall 4']
    for i in range(1, len(df_planning)):  # Start from the second task
        task = df_planning.iloc[i]
        prev_task = df_planning.iloc[i - 1]
        
        if task['Task'] not in masonry_subtasks:  # Only adjust non-overlapping tasks
            # Adjust start date based on the deviation of the previous task
            new_start_date = prev_task['End Date'] + pd.Timedelta(days=prev_task['Deviation'])
            
            # Ensure the start date doesn't violate the original planned start date
            df_planning.at[i, 'Start Date'] = max(task['Start Date'], new_start_date)
            df_planning.at[i, 'End Date'] = df_planning.at[i, 'Start Date'] + pd.Timedelta(days=task['Total Duration'])

    # Specifically adjust 'Plastering Wall' task
    plastering_task = df_planning[df_planning['Task'] == 'Plastering Wall'].iloc[0]
    prev_task = df_planning.iloc[df_planning[df_planning['Task'] == 'Masonry Wall 4'].index[0]]  # Assuming 'Plastering Wall' follows 'Masonry Wall 4'
    
    # Adjust 'Plastering Wall' based on deviation of 'Masonry Wall 4'
    new_start_date = prev_task['End Date'] + pd.Timedelta(days=prev_task['Deviation'])
    df_planning.at[plastering_task.name, 'Start Date'] = max(plastering_task['Start Date'], new_start_date)
    df_planning.at[plastering_task.name, 'End Date'] = df_planning.at[plastering_task.name, 'Start Date'] + pd.Timedelta(days=plastering_task['Total Duration'])

    # Reverse the order of tasks for visualization
    df_planning = df_planning[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, task in df_planning.iterrows():
        # Base bar for remaining duration
        ax.barh(task['Task'], task['Total Duration'], left=task['Start Date'], color="lightgray", label="Remaining Duration" if i == 0 else "")
        
        # Bar for current progress
        ax.barh(task['Task'], task['Current Progress Duration'], left=task['Start Date'], color="green", label="Current Progress" if i == 0 else "")
        
        # Bar for planned progress
        ax.barh(task['Task'], task['Planned Progress Duration'], left=task['Start Date'], color="none", edgecolor="black", hatch="//", label="Planned Progress" if i == 0 else "")
        
        # Visualization of deviation
        if task['Deviation'] > 0:  # Behind Schedule
            ax.barh(
                task['Task'],
                abs(task['Deviation']),
                left=task['End Date'],
                color="none",
                edgecolor="red",
                hatch="\\",
                label="Behind Schedule" if i == 0 else ""
            )
        elif task['Deviation'] < 0:  # Ahead of Schedule
            ax.barh(
                task['Task'],
                abs(task['Deviation']),
                left=task['End Date'] + pd.to_timedelta(task['Deviation'], unit='d'),
                color="white",
                edgecolor="blue",
                hatch="\\",
                label="Ahead of Schedule" if i == 0 else ""
            )

    ax.axvline(current_date, color='red', linestyle='--', label="Current Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Task")
    plt.title("Masonry Wall Progress Tracking")
    
    handles = [
        Patch(color='green', label='Current Progress'),
        Patch(edgecolor='black', hatch='//', label='Planned Progress', facecolor='none'),
        Patch(color= 'white', edgecolor='blue', hatch='\\', label='Ahead of Schedule', facecolor='none'),
        Patch(edgecolor='red', hatch='\\', label='Behind Schedule', facecolor='none'),
        Patch(color='lightgray', label='Remaining Duration')
    ]
    ax.legend(handles=handles, loc="upper right")
    plt.grid(True)
    
    return fig
