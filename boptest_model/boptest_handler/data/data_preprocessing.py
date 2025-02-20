import pandas as pd
import os
from datetime import datetime, timedelta

def get_script_directory():
    """Get the directory where the script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def find_csv_files(directory):
    """Find all CSV files in the directory and its subdirectories.
    
    Args:
        directory (str): Root directory to search for CSV files
        
    Returns:
        dict: Dictionary with base filenames as keys and lists of file paths as values
    """
    csv_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                base_name = os.path.splitext(os.path.basename(file))[0]
                file_path = os.path.join(root, file)
                if base_name in csv_files:
                    csv_files[base_name].append(file_path)
                else:
                    csv_files[base_name] = [file_path]
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

def create_timestamp_column(num_rows, start_date=datetime(2024, 1, 1)):
    """Create a timestamp column with 30-second intervals.
    
    Args:
        num_rows (int): Number of rows needed
        start_date (datetime): Starting date for timestamps
        
    Returns:
        list: List of timestamps
    """
    return [start_date + timedelta(seconds=i*30) for i in range(num_rows)]

def save_dataframe(df, output_path):
    """Save dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Output file path
    """
    df.to_csv(output_path, index=False)
    print(f"Created file: {output_path}")

def process_csv_files():
    """Main function to process all CSV files."""
    script_dir = get_script_directory()
    csv_files = find_csv_files(script_dir)
    
    # Group files by their base name
    df_dict = {}
    for base_name, file_paths in csv_files.items():
        # Process each CSV file
        for file_path in file_paths:
            df = read_and_process_dataframe(file_path)
            if base_name in df_dict:
                # Append to existing dataframe if base_name already exists
                df_dict[base_name] = pd.concat([df_dict[base_name], df], ignore_index=True)
            else:
                # Create new entry if base_name doesn't exist
                df_dict[base_name] = df
    
    # Create timestamps for each dataframe
    for base_name, df in df_dict.items():
        # Add timestamp column
        timestamps = create_timestamp_column(len(df))
        df.insert(0, 'timestamp', timestamps)
        # Create merged_data directory if it doesn't exist
        merged_data_dir = os.path.join(script_dir, "merged_data")
        os.makedirs(merged_data_dir, exist_ok=True)
        
        # Save processed dataframe
        output_path = os.path.join(merged_data_dir, f"{base_name}_processed.csv")
        save_dataframe(df, output_path)

if __name__ == "__main__":
    process_csv_files()