import pandas as pd
import os
import glob

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# List of files that need temperature conversion
files_to_convert = [
    'hvac_oveAhu_TSupSet_u_processed.csv',
    'hvac_reaAhu_TCooCoiRet_y_processed.csv',
    'hvac_reaAhu_TCooCoiSup_y_processed.csv',
    'hvac_reaAhu_THeaCoiRet_y_processed.csv',
    'hvac_reaAhu_THeaCoiSup_y_processed.csv',
    'hvac_reaAhu_TSup_y_processed.csv',
    'weaSta_reaWeaTWetBul_y_processed.csv',
]

# Process only the specified files
for file in files_to_convert:
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if file has at least 2 columns
        if len(df.columns) < 2:
            print(f"Skipping {file}: Less than 2 columns")
            continue
            
        print(f"Converting temperature in {file} from Kelvin to Celsius")
        
        # Convert the second column from K to °C
        df.iloc[:, 1] = df.iloc[:, 1].apply(kelvin_to_celsius)
        
        # Save the modified file
        df.to_csv(file, index=False)
        print(f"Saved modified {file}")
            
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("Processing complete!")
