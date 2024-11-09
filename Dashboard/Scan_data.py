import pandas as pd
import matplotlib.pyplot as plt

def plot_scan_data(df):
    '''
    Deze functie visualiseert de informatie over de totale tijd die besteed is aan scans, de duur van de huidige scan, 
    de gemiddelde duur van de scans, het aantal resterende scans, en de geschatte totale tijd voor de resterende scans. 
    Dit wordt weergegeven in een matplotlib-figuur met tekst.

    Parameters:
    df (pd.DataFrame): Een Pandas DataFrame die minimaal een kolom 'Duration' bevat waarin de duur van elke scan wordt 
                       opgeslagen (in seconden). 'Duration' moet een float of integer kolom zijn met eventuele lege 
                       waarden die als NaN kunnen worden behandeld.

    Returns:
    fig (matplotlib.figure.Figure): Een matplotlib-figuur waarin de samenvattende scaninformatie wordt weergegeven 
                                    als tekst in het midden van de figuur.
    '''

    # Bereken de totale tijd besteed aan eerdere scans (exclusief de laatste scan)
    if 'Duration' in df.columns and df['Duration'].notnull().any():
        total_time = df['Duration'].iloc[:-1].sum()
    else:
        total_time = 0

    # Bepaal de duur van de huidige scan (de laatste geldige invoer)
    current_scan_duration_index = df['Duration'].last_valid_index()
    current_scan_duration = df['Duration'].iloc[current_scan_duration_index] if current_scan_duration_index is not None else 0

    # Gemiddelde duur van alle scans (exclusief de laatste scan)
    average_duration = df['Duration'].mean() if 'Duration' in df.columns and df['Duration'].notnull().any() else 0

    # Resterende scans die moeten worden uitgevoerd
    remaining_scans = df.shape[0] - 1
    if current_scan_duration_index is not None:
        remaining_scans -= 1

    # Totale geschatte tijd om resterende scans te voltooien
    total_time_remaining = remaining_scans * average_duration if remaining_scans > 0 and average_duration > 0 else 0

    # Maak een display voor de resultaten
    fig, ax = plt.subplots(figsize=(6, 4))

    # Weergeef de informatie met een extra blanco regel
    display_text = (f"Total Time for Previous Scans: {total_time:.2f} seconds\n"
                     f"Current Scan Duration: {current_scan_duration:.2f} seconds\n"
                     f"Average Duration: {average_duration:.2f} seconds\n\n"  
                     f"Remaining Scans: {remaining_scans}\n"
                     f"Estimated Total Time for Remaining Scans: {total_time_remaining:.2f} seconds")

    ax.text(0.5, 0.5, display_text, fontsize=12, ha='center', va='center', color='#36607D')
    ax.set_axis_off()

    return fig
