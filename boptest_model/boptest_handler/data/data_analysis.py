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

# Example usage:
df = pd.read_csv('mix_day/hvac_reaAhu_TSup_y.csv')
time = df['time']
hvac_reaAhu_TSup_y = df['hvac_reaAhu_TSup_y']

df = pd.read_csv('mix_day/hvac_oveAhu_TSupSet_u.csv')
hvac_oveAhu_TSupSet_u = df['hvac_oveAhu_TSupSet_u']

plot_temperature_comparison(hvac_oveAhu_TSupSet_u, hvac_reaAhu_TSup_y, time)
