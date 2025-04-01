import os
import json
from pathlib import Path

def create_files_json(directory_path, output_json_path):
    """
    Create a JSON file with filenames as keys and empty values.
    
    Args:
        directory_path (str): Path to the directory containing the files
        output_json_path (str): Path where the JSON file should be saved
    """
    # Get all filenames in the directory
    files_dict = {}
    
    # Walk through directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            # Add filename as key with empty value
            files_dict[filename] = ""
    
    # Write to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(files_dict, json_file, indent=4)
    
    print(f"JSON file created at: {output_json_path}")
    print(f"Total files processed: {len(files_dict)}")

def create_multi_directory_json(directory_list, output_json_path):
    """
    Create a JSON file with multiple sections, each named after the last folder
    in the directory path, containing filenames as keys with empty values.
    
    Args:
        directory_list (list): List of directory paths to process
        output_json_path (str): Path where the JSON file should be saved
    """
    # Initialize main dictionary
    main_dict = {}
    
    # Process each directory
    for directory_path in directory_list:
        # Get the last folder name from the path
        folder_name = Path(directory_path).name
        
        # Initialize dictionary for this directory
        files_dict = {}
        
        # Walk through directory
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                # Add filename as key with empty value
                files_dict[filename] = ""
        
        # Add this directory's dict to main dict
        main_dict[folder_name] = files_dict
    
    # Write to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(main_dict, json_file, indent=4)
    
    print(f"Multi-directory JSON file created at: {output_json_path}")
    print(f"Total directories processed: {len(directory_list)}")
    for folder, files in main_dict.items():
        print(f"Files in {folder}: {len(files)}")

def update_component_files(base_path, data_paths_json):
    """
    Read data_paths.json and update component JSON files with the specified paths.
    
    Args:
        base_path (str): Base path to the model_parameters directory
        data_paths_json (str): Path to the data_paths.json file
    """
    # Read the data_paths.json file
    with open(data_paths_json, 'r') as f:
        data_paths = json.load(f)
    
    # Process each system type directory
    for system_type, files in data_paths.items():
        system_dir = os.path.join(base_path, system_type)
        
        # Process each file in the system directory
        for filename, filepath in files.items():
            if filepath:  # Only process if filepath is not empty
                component_json_path = os.path.join(system_dir, filename)
                
                # Check if the component JSON file exists
                if os.path.exists(component_json_path):
                    # Read the component JSON file
                    with open(component_json_path, 'r') as f:
                        component_data = json.load(f)
                    
                    # Update based on system type
                    if system_type == "SensorSystem":
                        component_data["readings"]["filename"] = filepath
                    
                    elif system_type == "ScheduleSystem":
                        component_data["readings"]["filename"] = filepath
                        component_data["parameters"]["useFile"] = True
                    
                    elif system_type == "OutdoorEnvironmentSystem":
                        component_data["readings"]["filename"] = filepath
                    
                    # Write the updated data back to the file
                    with open(component_json_path, 'w') as f:
                        json.dump(component_data, f, indent=4)
                    
                    print(f"Updated {filename} in {system_type}")
                else:
                    print(f"Warning: {filename} not found in {system_type}")

if __name__ == "__main__":
    """
    # Example usage for multiple directories
    directory_list = [
        # Use forward slashes instead of backslashes
        "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/five_rooms_only_template/model_parameters/SensorSystem",
        "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/five_rooms_only_template/model_parameters/ScheduleSystem",
        "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/five_rooms_only_template/model_parameters/OutdoorEnvironmentSystem"
    ]
    
    # Use os.path.join for constructing paths
    base_path = "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/five_rooms_only_template/model_parameters"
    multi_output_path = os.path.join(base_path, "data_paths.json")
    
    # Create the JSON files
    create_multi_directory_json(directory_list, multi_output_path)
    """
    base_path = "C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/five_rooms_only_template/model_parameters"
    # Update the component files
    data_paths_json = os.path.join(base_path, "data_paths.json")
    update_component_files(base_path, data_paths_json)