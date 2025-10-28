import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

def get_script_directory():
    """Get the directory where the script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def find_csv_files(directory):
    """Find all CSV files in the directory and its subdirectories, excluding merged_data.
    
    Args:
        directory (str): Root directory to search for CSV files
        
    Returns:
        dict: Dictionary with base filenames as keys and lists of tuples (file_path, folder_name) as values
    """
    csv_files = {}
    for root, dirs, files in os.walk(directory):
        # Skip the merged_data directory to avoid processing already processed files
        if 'merged_data' in root:
            continue
            
        for file in files:
            if file.endswith('.csv'):
                base_name = os.path.splitext(os.path.basename(file))[0]
                file_path = os.path.join(root, file)
                
                # Extract folder name from the path
                # Look for the main folder (mix_day, typical_cool_day, typical_heat_day)
                path_parts = os.path.normpath(file_path).split(os.sep)
                folder_name = None
                for part in path_parts:
                    if part in ['mix_day', 'typical_cool_day', 'typical_heat_day']:
                        folder_name = part
                        break
                
                # If no specific folder found, use 'default'
                if folder_name is None:
                    folder_name = 'default'
                
                if base_name in csv_files:
                    csv_files[base_name].append((file_path, folder_name))
                else:
                    csv_files[base_name] = [(file_path, folder_name)]
    return csv_files

def read_and_process_dataframe(file_path):
    """Read CSV file and remove the first column.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    df = pd.read_csv(file_path)
    return df.iloc[:, 1:]

def create_timestamp_column(num_rows, folder_name='default'):
    """Create a timestamp column with 30-second intervals.
    
    Args:
        num_rows (int): Number of rows needed
        folder_name (str): Name of the folder to determine start date
        
    Returns:
        list: List of timestamps
    """
    # Define start dates based on folder names
    start_dates = {
        'typical_heat_day': datetime(2024, 1, 11),  # 11th of January
        'mix_day': datetime(2024, 3, 17),           # 17th of March
        'typical_cool_day': datetime(2024, 5, 17),  # 17th of May
        'default': datetime(2024, 1, 1)             # Default fallback
    }
    
    start_date = start_dates.get(folder_name, start_dates['default'])
    return [start_date + timedelta(seconds=i*30) for i in range(num_rows)]

def save_dataframe(df, output_path):
    """Save dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Output file path
    """
    # Check if file already exists to indicate override
    file_exists = os.path.exists(output_path)
    df.to_csv(output_path, index=False)
    
    if file_exists:
        print(f"Overridden file: {output_path}")
    else:
        print(f"Created file: {output_path}")

def process_csv_files():
    """Main function to process all CSV files from original directories and merge files with same base name from different folders."""
    script_dir = get_script_directory()
    csv_files = find_csv_files(script_dir)
    
    # Group files by their base name only (not by folder)
    base_name_dict = {}
    for base_name, file_info_list in csv_files.items():
        if base_name not in base_name_dict:
            base_name_dict[base_name] = []
        base_name_dict[base_name].extend(file_info_list)
    
    # Process each base name and merge files from different folders
    for base_name, file_info_list in base_name_dict.items():
        merged_dfs = []
        
        # Define the order for concatenation: heat day, mix day, cool day
        folder_order = ['typical_heat_day', 'mix_day', 'typical_cool_day']
        
        # Process files in the specified order
        for folder_name in folder_order:
            for file_path, file_folder_name in file_info_list:
                if file_folder_name == folder_name:
                    df = read_and_process_dataframe(file_path)
                    
                    # Add timestamp column with folder-specific start date
                    timestamps = create_timestamp_column(len(df), folder_name)
                    df.insert(0, 'timestamp', timestamps)
                    
                    merged_dfs.append(df)
        
        # Concatenate all dataframes for this base name in the specified order
        if merged_dfs:
            final_df = pd.concat(merged_dfs, ignore_index=True)
            
            # Create merged_data directory if it doesn't exist
            merged_data_dir = os.path.join(script_dir, "merged_data")
            os.makedirs(merged_data_dir, exist_ok=True)
            
            # Save processed dataframe with original name (this will override existing files)
            output_path = os.path.join(merged_data_dir, f"{base_name}_processed.csv")
            save_dataframe(final_df, output_path)

def split_heating_and_cooling_setpoints(verbose=True):
    """Split the heating and cooling setpoints into separate files."""
    script_dir = get_script_directory()
    supply_temp_setpoint_dir = os.path.join(script_dir, "merged_data", "hvac_oveAhu_TSupSet_u_processed.csv")
    outdoor_temp_dir = os.path.join(script_dir, "merged_data", "weaSta_reaWeaTWetBul_y_processed.csv")
    
    # Read the supply temperature setpoint and outdoor temperature dataframes
    if verbose:
        print("Reading supply temperature setpoint and outdoor temperature data...")
    supply_temp_setpoint_df = pd.read_csv(supply_temp_setpoint_dir)
    outdoor_temp_df = pd.read_csv(outdoor_temp_dir)

    if verbose:
        print("Processing setpoints and splitting into heating and cooling...")
    
    # Create copies of the supply temp dataframe for heating and cooling
    heating_df = supply_temp_setpoint_df.copy()
    cooling_df = supply_temp_setpoint_df.copy()
    
    # Create a boolean mask for heating conditions (comparison in Kelvin)
    heating_mask = supply_temp_setpoint_df['hvac_oveAhu_TSupSet_u'] > outdoor_temp_df['weaSta_reaWeaTWetBul_y']
    
    # Set default values in Kelvin (12°C -> 285.15K, 26°C -> 299.15K)
    heating_df.loc[~heating_mask, 'hvac_oveAhu_TSupSet_u'] = 285.15  # 12°C in Kelvin
    cooling_df.loc[heating_mask, 'hvac_oveAhu_TSupSet_u'] = 299.15   # 26°C in Kelvin
    
    if verbose:
        heating_count = heating_mask.sum()
        cooling_count = (~heating_mask).sum()
        total_rows = len(supply_temp_setpoint_df)
        print(f"Processing complete. Found {heating_count} heating and {cooling_count} cooling setpoints.")
        print(f"Total rows processed: {total_rows}")
        print("Saving files...")

    # Save the heating and cooling dataframes
    heating_df.to_csv(os.path.join(script_dir, "merged_data", "heating_setpoints.csv"), index=False)
    cooling_df.to_csv(os.path.join(script_dir, "merged_data", "cooling_setpoints.csv"), index=False)

    if verbose:
        print("Files saved successfully!")

def clean_negative_values(df, column_name):
    """Clean negative values in a specific column of a dataframe."""
    df.loc[df[column_name] < 0, column_name] = 0
    return df

def clean_fan_power():
    """Clean negative values in the 'power' column of a dataframe."""
    script_dir = get_script_directory() 
    fan_power_dir = os.path.join(script_dir, "merged_data", "hvac_reaAhu_PFanSup_y_processed.csv")
    df = pd.read_csv(fan_power_dir)
    df = clean_negative_values(df, 'hvac_reaAhu_PFanSup_y')
    df.to_csv(os.path.join(script_dir, "merged_data", "hvac_reaAhu_PFanSup_y_processed.csv"), index=False)

def clean_airflow_rate():
    """Clean negative values in the 'airflow_rate' column of a dataframe."""
    script_dir = get_script_directory()
    airflow_rate_dir = os.path.join(script_dir, "merged_data", "hvac_reaAhu_V_flow_sup_y_processed.csv")
    df = pd.read_csv(airflow_rate_dir)
    df = clean_negative_values(df, 'hvac_reaAhu_V_flow_sup_y')
    df.to_csv(os.path.join(script_dir, "merged_data", "hvac_reaAhu_V_flow_sup_y_processed.csv"), index=False)

def process_occupancy_data(csv_file_path, save_csv=True):
    """
    Process occupancy data from a merged CSV file to create 30s intervals from 600s data.
    
    Args:
        csv_file_path (str): Path to the CSV file containing merged occupancy data
        
    Returns:
        pd.DataFrame: Processed occupancy data with 30s intervals
    """
    # Read the original CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the occupancy values only (excluding timestamp column)
    occupancy_values = df.iloc[:, 1:].values
    
    # Calculate how many 30s intervals we need for each 600s value (600/30 = 20)
    intervals_per_value = 20
    
    # Create expanded array by repeating each value 20 times
    expanded_values = np.repeat(occupancy_values, intervals_per_value, axis=0)
    
    # Add one more set of values (600s worth of 30s intervals) at the start
    first_values = occupancy_values[0]
    padding = np.tile(first_values, (intervals_per_value, 1))
    
    # Combine the padding with expanded values
    final_values = np.vstack([padding, expanded_values])
    
    # Create new timestamp array with 30s intervals
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    total_intervals = final_values.shape[0]
    timestamps = [start_time + pd.Timedelta(seconds=30*i) for i in range(total_intervals)]
    
    # Create new DataFrame with processed data
    result_df = pd.DataFrame(final_values, columns=df.columns[1:])
    result_df.insert(0, 'datetime', timestamps)
    
    if save_csv:
        #Saved in the same directory as the original file with the same name but with _30s_processed added
        output_path = os.path.splitext(csv_file_path)[0] + '_30s_processed.csv'
        result_df.to_csv(output_path, index=False)
        print(f"Processed occupancy data saved to {output_path}")
    
    return result_df

def process_all_occupancy_data():
    """Process all occupancy data files."""
    script_dir = get_script_directory()
    occupancy_files = ["Occupancy[cor]_processed.csv", 
                       "Occupancy[eas]_processed.csv", 
                       "Occupancy[nor]_processed.csv", 
                       "Occupancy[sou]_processed.csv", 
                       "Occupancy[wes]_processed.csv"]
    for file in occupancy_files:
        print(f"Processing occupancy data for {file}...")
        csv_file_path = os.path.join(script_dir, "merged_data", file)
        process_occupancy_data(csv_file_path, save_csv=True)
    print("All occupancy data processed successfully!")

def process_raw_occupancy_data(csv_file_path, save_csv=True):
    """
    Process raw occupancy data with integer timestamps to create 30s intervals 
    covering 2 weeks plus 600s before the start.
    
    Args:
        csv_file_path (str): Path to the CSV file containing raw occupancy data
        save_csv (bool): Whether to save the processed data to CSV
        
    Returns:
        pd.DataFrame: Processed occupancy data with 30s intervals
    """
    # Read the original CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the occupancy values (excluding timestamp column)
    occupancy_values = df.iloc[:, 1:].values
    
    # Get the initial timestamp
    initial_timestamp = df.iloc[0, 0]
    
    # Calculate how many 30s intervals we need for each 600s value
    intervals_per_value = 20  # 600/30 = 20
    
    # Create expanded array by repeating each value 20 times
    expanded_values = np.repeat(occupancy_values, intervals_per_value, axis=0)
    
    # Add padding at start using first values
    first_values = occupancy_values[0]
    padding = np.tile(first_values, (intervals_per_value, 1))
    
    # Combine padding with expanded values
    final_values = np.vstack([padding, expanded_values])
    
    # Create timestamp array covering (start-600) to (start+2weeks)
    start_time = initial_timestamp - 600
    end_time = start_time + 1209600  # 2 weeks = 1,209,600 seconds
    timestamps = np.arange(start_time, end_time + 30, 30) 
    
    # Ensure we have enough values to cover the full time range
    num_timestamps = len(timestamps)
    if final_values.shape[0] < num_timestamps:
        # Calculate how many times to repeat the pattern
        repeats_needed = (num_timestamps - final_values.shape[0]) // final_values.shape[0] + 1
        final_values = np.tile(final_values, (repeats_needed, 1))[:num_timestamps]
    else:
        # Trim excess values
        final_values = final_values[:num_timestamps]
    
    #Convert the integers to floats
    final_values = final_values.astype(float)

    # Create new DataFrame
    result_df = pd.DataFrame(final_values, columns=df.columns[1:])
    result_df.insert(0, 'timestamp', timestamps)
    
    if save_csv:
        # Save in the same directory with '_30s_processed.csv' suffix
        output_path = os.path.splitext(csv_file_path)[0] + '_30s.csv'
        result_df.to_csv(output_path, index=False)
        print(f"Processed raw occupancy data saved to {output_path}")
    
    return result_df

def process_all_raw_occupancy_data():
    """Process all raw occupancy data files."""
    script_dir = get_script_directory()
    mix_day_dir = os.path.join(script_dir, "mix_day/forecasts")
    typical_cool_day_dir = os.path.join(script_dir, "typical_cool_day/forecasts")
    typical_heat_day_dir = os.path.join(script_dir, "typical_heat_day/forecasts")   
    file_names = ["UpperCO2[cor].csv", "UpperCO2[eas].csv", "UpperCO2[nor].csv", "UpperCO2[sou].csv", "UpperCO2[wes].csv"]
    for file_name in file_names:
        process_raw_occupancy_data(os.path.join(mix_day_dir, file_name), save_csv=True)
        process_raw_occupancy_data(os.path.join(typical_cool_day_dir, file_name), save_csv=True)
        process_raw_occupancy_data(os.path.join(typical_heat_day_dir, file_name), save_csv=True)
    print("All raw occupancy data processed successfully!")

def create_outdoor_env_data():
    """Create outdoor environment data."""
    script_dir = get_script_directory()
    outdoorTemperature = os.path.join(script_dir, "merged_data", "weaSta_reaWeaTWetBul_y_processed.csv")
    globalIrradiation = os.path.join(script_dir, "merged_data", "weaSta_reaWeaHGloHor_y_processed.csv")

    outdoorTemperature_df = pd.read_csv(outdoorTemperature)

    # Convert outdoorTemperature to Celsius
    outdoorTemperature_df['weaSta_reaWeaTWetBul_y'] = (outdoorTemperature_df['weaSta_reaWeaTWetBul_y'] - 273.15)

    globalIrradiation_df = pd.read_csv(globalIrradiation)

    # Create a new DataFrame with the merged data
    merged_df = pd.DataFrame({
        'timestamp': outdoorTemperature_df['timestamp'],
        'outdoorTemperature': outdoorTemperature_df['weaSta_reaWeaTWetBul_y'],
        'globalIrradiation': globalIrradiation_df['weaSta_reaWeaHGloHor_y']
    })


    merged_df.to_csv(os.path.join(script_dir, "merged_data", "outdoor_env_data.csv"), index=False)
    print("Outdoor environment data created successfully!")

def create_water_temperature_data():
    """Create water temperature data."""
    script_dir = get_script_directory()
    main_coil_inlet_water_temperature = os.path.join(script_dir, "merged_data", "hvac_reaAhu_THeaCoiSup_y_processed.csv")
    #Create two dataframes, one for the inlet and one for the outlet with the same size
    # and timestamp as the main_coil_inlet_water_temperature, then fill the inlet with constant 45 degrees
    # and the outlet with constant 35 degrees
    main_coil_inlet_water_temperature = pd.read_csv(main_coil_inlet_water_temperature)

    #Create two dataframes, one for the inlet and one for the outlet with the same size
    coils_inlet_water_temperature = pd.DataFrame(index=main_coil_inlet_water_temperature.index)
    coils_outlet_water_temperature = pd.DataFrame(index=main_coil_inlet_water_temperature.index)

    #Keep the timestamp from the main_coil_inlet_water_temperature
    coils_inlet_water_temperature['timestamp'] = main_coil_inlet_water_temperature['timestamp']
    coils_outlet_water_temperature['timestamp'] = main_coil_inlet_water_temperature['timestamp']

    coils_inlet_water_temperature['inlet_water_temperature'] = 45
    coils_outlet_water_temperature['outlet_water_temperature'] = 35

    #Save the two dataframes to csv
    coils_inlet_water_temperature.to_csv(os.path.join(script_dir, "merged_data", "coils_inlet_water_temperature.csv"), index=False)
    coils_outlet_water_temperature.to_csv(os.path.join(script_dir, "merged_data", "coils_outlet_water_temperature.csv"), index=False)
    print("Water temperature data created successfully!")

def create_airflow_based_damper_positions():
    """Read all the zone airflow data and create a damper position based on the airflow rate.
    - Normalize the airflow rate to the maximum airflow rate in the dataframe
    - Save the damper position to a csv file with timestamp from original data
    """
    script_dir = get_script_directory()
    zone_airflow_files = ["hvac_reaZonCor_V_flow_y_processed.csv",
                         "hvac_reaZonEas_V_flow_y_processed.csv",
                         "hvac_reaZonNor_V_flow_y_processed.csv",
                         "hvac_reaZonSou_V_flow_y_processed.csv",
                         "hvac_reaZonWes_V_flow_y_processed.csv"]
    
    for file in zone_airflow_files:
        df = pd.read_csv(os.path.join(script_dir, "merged_data", file))
        #get the name of the second column
        column_name = df.columns[1]
        #Normalize the airflow rate to the maximum airflow rate in the dataframe
        max_airflow_rate = df[column_name].max()
        df[column_name] = df[column_name] / max_airflow_rate

        #Create a new dataframe with timestamp and damper position
        airflow_based_damper_position = pd.DataFrame({
            'timestamp': df['timestamp'],
            'damper_position': df[column_name]
        })

        #Save the damper position to a csv file
        airflow_based_damper_position.to_csv(os.path.join(script_dir, "merged_data", file.replace("_V_flow_y_processed.csv", "_damper_position_y_processed.csv")), index=False)
        
        print(f"Airflow-based damper position data created for {file}")

def transform_temp_to_celsius(csv_file_path):
    """Read CSV file and transform temperature data from Kelvin to Celsius."""
    df = pd.read_csv(csv_file_path)
    #Temperature data is always the second column

    #Assert that the temperature data is in Kelvin by checking the average value is above 220
    assert df.iloc[:, 1].mean() > 220, "Temperature data is not in Kelvin"
    
    df.iloc[:, 1] = df.iloc[:, 1] - 273.15
    df.to_csv(csv_file_path, index=False)

def transform_temp_to_celsius_all():
    """Read all CSV files and transform temperature data from Kelvin to Celsius."""
    script_dir = get_script_directory()
    temp_files = ["weaSta_reaWeaTWetBul_y_processed.csv", 
                  "hvac_oveAhu_TSupSet_u_processed.csv",
                  "hvac_oveZonSupCor_TZonCooSet_u_processed.csv",
                  "hvac_oveZonSupCor_TZonHeaSet_u_processed.csv",
                  "hvac_oveZonSupEas_TZonCooSet_u_processed.csv",
                  "hvac_oveZonSupEas_TZonHeaSet_u_processed.csv",
                  "hvac_oveZonSupNor_TZonCooSet_u_processed.csv",
                  "hvac_oveZonSupNor_TZonHeaSet_u_processed.csv",
                  "hvac_oveZonSupSou_TZonCooSet_u_processed.csv",
                  "hvac_oveZonSupSou_TZonHeaSet_u_processed.csv",
                  "hvac_oveZonSupWes_TZonCooSet_u_processed.csv",
                  "hvac_oveZonSupWes_TZonHeaSet_u_processed.csv",
                  "hvac_reaAhu_TCooCoiRet_y_processed.csv",
                  "hvac_reaAhu_TCooCoiSup_y_processed.csv",
                  "hvac_reaAhu_THeaCoiRet_y_processed.csv",
                  "hvac_reaAhu_THeaCoiSup_y_processed.csv",
                  "hvac_reaZonCor_TSup_y_processed.csv",
                  "hvac_reaZonCor_TZon_y_processed.csv",
                  "hvac_reaZonEas_TSup_y_processed.csv",
                  "hvac_reaZonEas_TZon_y_processed.csv",
                  "hvac_reaZonNor_TSup_y_processed.csv",
                  "hvac_reaZonNor_TZon_y_processed.csv",
                  "hvac_reaZonSou_TSup_y_processed.csv",
                  "hvac_reaZonSou_TZon_y_processed.csv",
                  "hvac_reaZonWes_TSup_y_processed.csv",
                  "hvac_reaZonWes_TZon_y_processed.csv",
                  "hvac_reaAhu_TSup_y_processed.csv",
                  "hvac_reaAhu_TRet_y_processed.csv",
                  ]
    for file in temp_files:
        print(f"Transforming temperature data for {file}...")
        transform_temp_to_celsius(os.path.join(script_dir, "merged_data", file))
    print("All temperature data transformed successfully!")

def regenerate_all_data():
    process_csv_files()
    process_all_occupancy_data()
    create_outdoor_env_data()
    create_water_temperature_data()
    create_airflow_based_damper_positions()
    transform_temp_to_celsius_all()

if __name__ == "__main__":
    regenerate_all_data()