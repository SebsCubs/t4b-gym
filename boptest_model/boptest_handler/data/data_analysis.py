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
        print(plot_df)
    
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

# Example usage:

# For AHU information
ahu_data = {
    'AHU differential pressure setpoint': 'hvac_oveAhu_dpSet_u',
    'AHU temp setpoint': 'hvac_oveAhu_TSupSet_u',
    'AHU FAN speed': 'hvac_oveAhu_yFan_u'
}
plot_timeseries_data(ahu_data, 'typical_heat_day')


