import pandas as pd
import os
import glob

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# Get all CSV files in the current directory
csv_files = glob.glob('*.csv')

for file in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if file has at least 2 columns
        if len(df.columns) < 2:
            print(f"Skipping {file}: Less than 2 columns")
            continue
            
        # Get the second column
        second_col = df.iloc[:, 1]
        
        # Check if values are within 250-320 K range
        if second_col.between(250, 320).any():
            print(f"Converting temperature in {file} from Kelvin to Celsius")
            
            # Convert the second column from K to °C
            df.iloc[:, 1] = df.iloc[:, 1].apply(kelvin_to_celsius)
            
            # Save the modified file
            df.to_csv(file, index=False)
            print(f"Saved modified {file}")
        else:
            print(f"Skipping {file}: No values between 250-320 K in second column")
            
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("Processing complete!")
