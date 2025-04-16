import pandas as pd
import matplotlib.pyplot as plt

# Assuming the data is already loaded into these variables
# If not, you'll need to load your data first
def plot_temperature_comparison(hvac_oveAhu_TSupSet_u, hvac_reaAhu_TSup_y, time):
    # Create a DataFrame with the time and temperature data
    df = pd.DataFrame({
        'time': time,
        'setpoint': hvac_oveAhu_TSupSet_u,
        'measured': hvac_reaAhu_TSup_y
    })
    
    # Convert time to datetime (assuming the start time is at 950400 seconds)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    
    # Resample to 600-second intervals using mean
    df_resampled = df.resample('600S').mean()
    
    # Convert temperatures from Kelvin to Celsius
    df_resampled['setpoint_celsius'] = df_resampled['setpoint'] - 273.15
    df_resampled['measured_celsius'] = df_resampled['measured'] - 273.15
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['setpoint_celsius'], 
             label='Temperature Setpoint', linewidth=2)
    plt.plot(df_resampled.index, df_resampled['measured_celsius'], 
             label='Measured Temperature', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Supply Air Temperature: Setpoint vs Measured')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()


def plot_supp_ret_dampers(hvac_oveAhu_yOA_u, hvac_oveAhu_yRet_u, time):
    # Create a DataFrame with the time and temperature data
    df = pd.DataFrame({
        'time': time,
        'OA': hvac_oveAhu_yOA_u,
        'Ret': hvac_oveAhu_yRet_u
    })
    
    # Convert time to datetime (assuming the start time is at 950400 seconds)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    
    # Resample to 600-second intervals using mean
    df_resampled = df.resample('600S').mean()
    
    # Convert temperatures from Kelvin to Celsius
    df_resampled['OA'] = df_resampled['OA']
    df_resampled['Ret'] = df_resampled['Ret']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['OA'], 
             label='OA', linewidth=2)
    plt.plot(df_resampled.index, df_resampled['Ret'], 
             label='Ret', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Damper Position')
    plt.title('Supply and Return Dampers')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()


def plot_timeseries_data(data_dict, scenario, resample_interval='600S', figsize=(12, 6)):
    """
    Plot multiple time series from CSV files in a specified scenario folder.
    
    Args:
        data_dict (dict): Dictionary with display names as keys and CSV file identifiers as values
                         e.g., {'Outside Air Damper': 'hvac_oveAhu_yOA_u'}
        scenario (str): Name of the folder containing the CSV files (e.g., 'typical_heat_day')
        resample_interval (str): Pandas resample interval (default: '600S' for 10 minutes)
        figsize (tuple): Figure size in inches (default: (12, 6))
    """
    # Initialize DataFrame with time from the first CSV
    first_file = list(data_dict.values())[0]
    df = pd.read_csv(f'{scenario}/{first_file}.csv')
    time = df['time']
    # Create initial DataFrame with time
    plot_df = pd.DataFrame({'time': time})
    #set the time as the index
    plot_df.set_index('time', inplace=True)
    
    # Add each data series to the DataFrame
    for display_name, file_id in data_dict.items():
        df = pd.read_csv(f'{scenario}/{file_id}.csv')
        df.set_index('time', inplace=True)
        plot_df[display_name] = df[file_id]

    
    # Convert time to datetime (starting from January 1st, 2006)
    #The current index is the time in seconds since the beggining of the year, we need to convert it to a datetime object   
    start_date = pd.Timestamp('2006-01-01')
    plot_df['datetime'] = start_date + pd.to_timedelta(plot_df.index, unit='s')
    plot_df.set_index('datetime', inplace=True)
    # Resample data
    df_resampled = plot_df.resample(resample_interval).mean()
    
    

    # Create separate plots for each data series
    for column in df_resampled.columns:
        plt.figure(figsize=figsize)
        plt.plot(df_resampled.index, df_resampled[column], 
                linewidth=2, color='blue')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{column} - {scenario}')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        plt.show()



def plot_merged_data(list_of_files, list_of_labels):
    """
    Plot multiple time series from CSV files in the same plot with interactive legend
    
    Args:
        list_of_files (list): List of paths to CSV files to plot
        list_of_labels (list): List of labels for the legend
    """
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Store the line objects and their visibility state
    lines = []
    
    # Loop through the list of files
    for i, file in enumerate(list_of_files):
        # Read the CSV file
        df = pd.read_csv(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        # Get the actual data column name (should be the only other column besides timestamp)
        data_column = df.columns[0]
        
        # Plot the data and store the line object
        line, = ax.plot(df.index, df[data_column], label=list_of_labels[i])
        lines.append(line)
    
    # Create the legend
    leg = ax.legend()
    
    # Make the legend interactive
    lined = {}  # Will map legend lines to original lines
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line
        lined[legline] = origline
    
    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend proxy line
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see which lines have been toggled
        legline.set_alpha(1.0 if vis else 0.2)
        fig.canvas.draw()
    
    # Connect the pick event to the on_pick function
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Merged Data (Click legend to toggle lines)')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":

    """
    # For AHU information
    ahu_data = {
        'AHU differential pressure setpoint': 'hvac_oveAhu_dpSet_u',
        'AHU temp setpoint': 'hvac_oveAhu_TSupSet_u',
        'AHU FAN speed': 'hvac_oveAhu_yFan_u',
        'AHU OA damper': 'hvac_oveAhu_yOA_u',
        'AHU Ret damper': 'hvac_oveAhu_yRet_u',
        'AHU Heater': 'hvac_oveAhu_yHea_u',
        'AHU Cooler': 'hvac_oveAhu_yCoo_u',
        'AHU Heater activate': 'hvac_oveAhu_yPumCoo_u',
        'AHU Cooler activate': 'hvac_oveAhu_yPumHea_u'
    }

    ahu_data = {
        'AHU Supply Air Temperature': 'hvac_reaAhu_TSup_y',
        'AHU Supply Air Flow': 'hvac_reaAhu_V_flow_sup_y',
        'Weather Station Wet Bulb Temperature': 'weaSta_reaWeaTWetBul_y',
        'AHU Supply Air Temperature Setpoint': 'hvac_oveAhu_TSupSet_u',
        'AHU Supply Fan Power': 'hvac_reaAhu_PFanSup_y',
        'AHU Heating Coil Supply Temperature': 'hvac_reaAhu_THeaCoiSup_y',
        'AHU Heating Coil Return Temperature': 'hvac_reaAhu_THeaCoiRet_y',
        'AHU Cooling Coil Supply Temperature': 'hvac_reaAhu_TCooCoiSup_y',
        'AHU Cooling Coil Return Temperature': 'hvac_reaAhu_TCooCoiRet_y',
        'AHU Outside Air Damper Position': 'hvac_oveAhu_yOA_u',
        'AHU Return Air Damper Position': 'hvac_oveAhu_yRet_u',
        'AHU Heating Valve Position': 'hvac_oveAhu_yHea_u',
        'AHU Cooling Valve Position': 'hvac_oveAhu_yCoo_u',
        'Zone Temperature': 'hvac_reaZonCor_TZon_y'
    }

    plot_timeseries_data(ahu_data, 'typical_heat_day')
    """
    # Plot the merged data
    list_of_files = ["C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_THeaCoiSup_y_processed.csv",
                    "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TCooCoiSup_y_processed.csv"]
    list_of_labels = ["Heating coil supply temperature", "Cooling coil supply temperature"]
    plot_merged_data(list_of_files, list_of_labels)
