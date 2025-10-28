import numpy as np
import os
import sys
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime

class TeeOutput:
    """Class to duplicate output to both console and file."""
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def setup_logging(log_file_path: str) -> TeeOutput:
    """Setup logging to both console and file."""
    tee = TeeOutput(log_file_path)
    sys.stdout = tee
    return tee

def analyze_npz_structure(file_path: str) -> Dict[str, Any]:
    """
    Analyze the structure of an NPZ file containing expert trajectories.
    
    Args:
        file_path: Path to the NPZ file
        
    Returns:
        Dictionary containing analysis results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"\n=== Analyzing {file_path} ===")
    
    # Load the NPZ file (allow_pickle=True for object arrays)
    data = np.load(file_path, allow_pickle=True)
    
    analysis = {
        'file_path': file_path,
        'keys': list(data.keys()),
        'shapes': {},
        'dtypes': {},
        'sample_data': {},
        'total_size_mb': 0
    }
    
    # Analyze each array in the file
    for key in data.keys():
        array = data[key]
        analysis['shapes'][key] = array.shape
        analysis['dtypes'][key] = array.dtype
        
        # Calculate size in MB
        size_mb = array.nbytes / (1024 * 1024)
        analysis['total_size_mb'] += size_mb
        
        # Store sample data (first few elements)
        try:
            if array.ndim == 1:
                sample_size = min(5, len(array))
                if array.dtype == object:
                    # For object arrays, show type info and first element
                    analysis['sample_data'][key] = {
                        'type': 'object_array',
                        'length': len(array),
                        'first_element_type': type(array[0]).__name__ if len(array) > 0 else 'empty',
                        'first_element': str(array[0])[:100] if len(array) > 0 else None  # Truncate long strings
                    }
                else:
                    analysis['sample_data'][key] = array[:sample_size].tolist()
            elif array.ndim == 2:
                sample_size = min(5, array.shape[0])
                if array.dtype == object:
                    analysis['sample_data'][key] = {
                        'type': 'object_array_2d',
                        'shape': array.shape,
                        'first_element_type': type(array[0, 0]).__name__ if array.size > 0 else 'empty',
                        'first_element': str(array[0, 0])[:100] if array.size > 0 else None
                    }
                else:
                    analysis['sample_data'][key] = array[:sample_size].tolist()
            else:
                # For higher dimensional arrays, show shape and first element
                analysis['sample_data'][key] = {
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'first_element': str(array.flat[0])[:100] if array.size > 0 else None
                }
        except Exception as e:
            # If we can't access the data, just store basic info
            analysis['sample_data'][key] = {
                'error': f"Could not access sample data: {str(e)}",
                'shape': array.shape,
                'dtype': str(array.dtype)
            }
        
        print(f"Key: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Dtype: {array.dtype}")
        print(f"  Size: {size_mb:.2f} MB")
        if array.ndim <= 2:
            print(f"  Sample: {analysis['sample_data'][key]}")
        print()
    
    print(f"Total file size: {analysis['total_size_mb']:.2f} MB")
    return analysis

def compare_trajectory_structures(analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the structures of two trajectory files to determine compatibility.
    
    Args:
        analysis1: Analysis results from first file
        analysis2: Analysis results from second file
        
    Returns:
        Dictionary containing compatibility analysis
    """
    print("\n=== Compatibility Analysis ===")
    
    comparison = {
        'common_keys': [],
        'unique_to_file1': [],
        'unique_to_file2': [],
        'compatible_keys': [],
        'incompatible_keys': [],
        'shape_differences': {},
        'dtype_differences': {}
    }
    
    keys1 = set(analysis1['keys'])
    keys2 = set(analysis2['keys'])
    
    # Find common and unique keys
    comparison['common_keys'] = list(keys1.intersection(keys2))
    comparison['unique_to_file1'] = list(keys1 - keys2)
    comparison['unique_to_file2'] = list(keys2 - keys1)
    
    print(f"Common keys: {comparison['common_keys']}")
    print(f"Unique to file 1: {comparison['unique_to_file1']}")
    print(f"Unique to file 2: {comparison['unique_to_file2']}")
    
    # Analyze compatibility of common keys
    for key in comparison['common_keys']:
        shape1 = analysis1['shapes'][key]
        shape2 = analysis2['shapes'][key]
        dtype1 = analysis1['dtypes'][key]
        dtype2 = analysis2['dtypes'][key]
        
        # Check if shapes are compatible (same number of dimensions, compatible sizes)
        compatible = True
        issues = []
        
        if len(shape1) != len(shape2):
            compatible = False
            issues.append(f"Different number of dimensions: {len(shape1)} vs {len(shape2)}")
        else:
            # Check each dimension
            for i, (s1, s2) in enumerate(zip(shape1, shape2)):
                if i == 0:  # First dimension (usually batch size) can be different
                    continue
                if s1 != s2:
                    compatible = False
                    issues.append(f"Dimension {i} differs: {s1} vs {s2}")
        
        # Check dtype compatibility
        if dtype1 != dtype2:
            issues.append(f"Different dtypes: {dtype1} vs {dtype2}")
            # Still compatible if we can convert
        
        if compatible:
            comparison['compatible_keys'].append(key)
            print(f"✓ {key}: Compatible")
        else:
            comparison['incompatible_keys'].append(key)
            comparison['shape_differences'][key] = issues
            print(f"✗ {key}: Incompatible - {', '.join(issues)}")
    
    return comparison

def merge_trajectory_files(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    Merge two trajectory files based on compatibility analysis.
    
    Args:
        file1_path: Path to first trajectory file
        file2_path: Path to second trajectory file
        output_path: Path for the merged output file
    """
    print(f"\n=== Merging Trajectory Files ===")
    
    # Load both files (allow_pickle=True for object arrays)
    data1 = np.load(file1_path, allow_pickle=True)
    data2 = np.load(file2_path, allow_pickle=True)
    
    # Analyze structures
    analysis1 = analyze_npz_structure(file1_path)
    analysis2 = analyze_npz_structure(file2_path)
    
    # Compare structures
    comparison = compare_trajectory_structures(analysis1, analysis2)
    
    # Create merged data
    merged_data = {}
    
    # Handle compatible keys (concatenate along first dimension)
    for key in comparison['compatible_keys']:
        array1 = data1[key]
        array2 = data2[key]
        
        try:
            # Concatenate along first dimension
            merged_array = np.concatenate([array1, array2], axis=0)
            merged_data[key] = merged_array
            
            print(f"Merged {key}: {array1.shape} + {array2.shape} = {merged_array.shape}")
        except Exception as e:
            print(f"Warning: Could not concatenate {key}: {str(e)}")
            # Try to merge as lists if concatenation fails
            try:
                if array1.ndim == 1 and array2.ndim == 1:
                    merged_list = list(array1) + list(array2)
                    merged_data[key] = np.array(merged_list, dtype=object)
                    print(f"Force-merged {key} as list: {len(merged_list)} elements")
                else:
                    print(f"Error: Cannot merge {key} - incompatible for list merging")
            except Exception as e2:
                print(f"Error: Complete failure to merge {key}: {str(e2)}")
    
    # Handle unique keys (include from both files)
    # Special handling for obs_next/next_obs merge
    obs_next_merged = False
    
    for key in comparison['unique_to_file1']:
        if key == 'obs_next':
            # Check if file 2 has 'next_obs' to merge with
            if 'next_obs' in comparison['unique_to_file2']:
                try:
                    array1 = data1['obs_next']
                    array2 = data2['next_obs']
                    
                    # Try to concatenate the arrays
                    merged_array = np.concatenate([array1, array2], axis=0)
                    merged_data['obs_next'] = merged_array
                    obs_next_merged = True
                    
                    print(f"Merged obs_next + next_obs: {array1.shape} + {array2.shape} = {merged_array.shape}")
                except Exception as e:
                    print(f"Warning: Could not merge obs_next + next_obs: {str(e)}")
                    # Fallback to list merging
                    try:
                        if array1.ndim == 1 and array2.ndim == 1:
                            merged_list = list(array1) + list(array2)
                            merged_data['obs_next'] = np.array(merged_list, dtype=object)
                            obs_next_merged = True
                            print(f"Force-merged obs_next + next_obs as list: {len(merged_list)} elements")
                        else:
                            # Just use the first one if merging fails
                            merged_data[key] = data1[key]
                            print(f"Added obs_next from file 1 (merge failed): {data1[key].shape}")
                    except Exception as e2:
                        merged_data[key] = data1[key]
                        print(f"Added obs_next from file 1 (merge failed): {data1[key].shape}")
            else:
                merged_data[key] = data1[key]
                print(f"Added unique key from file 1: {key} with shape {data1[key].shape}")
        else:
            merged_data[key] = data1[key]
            print(f"Added unique key from file 1: {key} with shape {data1[key].shape}")
    
    for key in comparison['unique_to_file2']:
        if key == 'next_obs' and obs_next_merged:
            # Skip this key since we already merged it with obs_next
            print(f"Skipped next_obs (already merged with obs_next)")
        else:
            merged_data[key] = data2[key]
            print(f"Added unique key from file 2: {key} with shape {data2[key].shape}")
    
    # Handle incompatible keys (try to convert and merge)
    for key in comparison['incompatible_keys']:
        array1 = data1[key]
        array2 = data2[key]
        
        # Try to make them compatible
        if len(array1.shape) == len(array2.shape):
            # Same number of dimensions, try to pad or truncate
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(array1.shape, array2.shape))
            
            # Truncate both arrays to minimum shape
            if array1.ndim == 1:
                truncated1 = array1[:min_shape[0]]
                truncated2 = array2[:min_shape[0]]
            elif array1.ndim == 2:
                truncated1 = array1[:min_shape[0], :min_shape[1]]
                truncated2 = array2[:min_shape[0], :min_shape[1]]
            else:
                # For higher dimensions, use advanced indexing
                indices = tuple(slice(0, s) for s in min_shape)
                truncated1 = array1[indices]
                truncated2 = array2[indices]
            
            # Convert to same dtype if needed
            if array1.dtype != array2.dtype:
                # Convert to the more general type
                if np.issubdtype(array1.dtype, np.integer) and np.issubdtype(array2.dtype, np.integer):
                    target_dtype = np.promote_types(array1.dtype, array2.dtype)
                elif np.issubdtype(array1.dtype, np.floating) and np.issubdtype(array2.dtype, np.floating):
                    target_dtype = np.promote_types(array1.dtype, array2.dtype)
                else:
                    target_dtype = np.float64  # Default to float64
                
                truncated1 = truncated1.astype(target_dtype)
                truncated2 = truncated2.astype(target_dtype)
            
            merged_array = np.concatenate([truncated1, truncated2], axis=0)
            merged_data[key] = merged_array
            
            print(f"Force-merged {key}: {array1.shape} + {array2.shape} -> {merged_array.shape} (truncated)")
        else:
            print(f"Warning: Cannot merge {key} - incompatible dimensions")
    
    # Save merged data
    np.savez(output_path, **merged_data)
    
    # Calculate final statistics
    total_size = sum(arr.nbytes for arr in merged_data.values()) / (1024 * 1024)
    print(f"\nMerged file saved to: {output_path}")
    print(f"Total merged size: {total_size:.2f} MB")
    print(f"Number of arrays: {len(merged_data)}")
    
    # Save merge report
    report = {
        'file1': file1_path,
        'file2': file2_path,
        'output': output_path,
        'merge_timestamp': np.datetime64('now').astype(str),
        'analysis1': analysis1,
        'analysis2': analysis2,
        'comparison': comparison,
        'merged_keys': list(merged_data.keys()),
        'merged_shapes': {k: v.shape for k, v in merged_data.items()}
    }
    
    report_path = output_path.replace('.npz', '_merge_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Merge report saved to: {report_path}")

def identify_temperature_indices(data: np.ndarray, signal_name: str = "unknown", sample_size: int = 1000) -> List[int]:
    """
    Identify which indices in the second dimension correspond to temperature signals.
    
    Args:
        data: The data array to analyze (shape: N, M where N=samples, M=signals)
        signal_name: Name of the signal for debugging
        sample_size: Number of samples to use for analysis (to speed up processing)
        
    Returns:
        List of indices that correspond to temperature signals
    """
    try:
        if data.ndim != 2:
            print(f"  {signal_name}: Expected 2D array, got {data.ndim}D")
            return []
        
        n_samples, n_signals = data.shape
        
        # Use a sample of the data for efficiency
        sample_n = min(sample_size, n_samples)
        sample_data = data[:sample_n, :]
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(sample_data)
        
        temperature_indices = []
        
        print(f"  {signal_name}: Analyzing {n_signals} signals using {sample_n} samples...")
        
        # Analyze each signal (column) across all samples
        for i in range(n_signals):
            signal_data = sample_data[:, i]
            valid_signal_data = signal_data[valid_mask[:, i]]
            
            if len(valid_signal_data) == 0:
                continue
            
            # Calculate mean and variance for this signal
            mean_val = np.mean(valid_signal_data)
            variance_val = np.var(valid_signal_data)
            
            # Temperature in Kelvin criteria:
            # - Mean should be around 273-350K (0-77°C)
            # - Variance should be relatively low (< 50) for temperature
            is_temperature = (mean_val > 270 and mean_val < 350 and variance_val < 60)
            
            if is_temperature:
                temperature_indices.append(i)
                print(f"    Signal {i}: mean={mean_val:.2f}, variance={variance_val:.2f} -> TEMPERATURE")
            else:
                print(f"    Signal {i}: mean={mean_val:.2f}, variance={variance_val:.2f}")
        
        #print(f"  {signal_name}: Found {len(temperature_indices)} temperature signals at indices: {temperature_indices}")
        return temperature_indices
        
    except Exception as e:
        print(f"  {signal_name}: Error analyzing - {str(e)}")
        return []

def convert_boptest_temperatures(input_file: str, output_file: str) -> None:
    """
    Convert temperature values from Kelvin to Celsius in boptest trajectory file.
    
    Args:
        input_file: Path to the input boptest NPZ file
        output_file: Path for the output NPZ file with converted temperatures
    """
    print(f"\n=== Converting Boptest Temperatures ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the NPZ file
    data = np.load(input_file, allow_pickle=True)
    
    converted_data = {}
    temperature_indices = {}
    
    print("\nAnalyzing signals for temperature identification:")
    
    # First pass: Identify temperature indices for each data type
    obs_temp_indices = None
    acts_temp_indices = None
    
    # Analyze each array to identify temperature indices
    for key in data.keys():
        array = data[key]
        print(f"\nAnalyzing {key} (shape: {array.shape}):")
        
        if array.dtype == object:
            # Handle object arrays (likely containing lists of observations/actions)
            if array.ndim == 1:
                # 1D object array - each element might be a list of values
                # First, identify temperature indices from a sample
                sample_elements = array[:min(100, len(array))]  # Sample first 100 elements
                sample_arrays = []
                
                for element in sample_elements:
                    if isinstance(element, (list, np.ndarray)):
                        element_array = np.array(element)
                        if element_array.ndim == 1:
                            sample_arrays.append(element_array)
                
                if sample_arrays:
                    # Stack sample arrays to create a 2D array for analysis
                    sample_data = np.stack(sample_arrays)
                    temp_indices = identify_temperature_indices(sample_data, f"{key}_sample")
                    
                    # Store indices based on data type
                    if 'obs' in key.lower():
                        obs_temp_indices = temp_indices
                        #print(f"  Stored observation temperature indices: {temp_indices}")
                    elif 'act' in key.lower():
                        acts_temp_indices = temp_indices
                        print(f"  Stored action temperature indices: {temp_indices}")
                    
                    temperature_indices[key] = temp_indices
            
        else:
            # Regular numpy array
            if array.ndim == 2:
                # 2D array - analyze each column (signal) across all rows (samples)
                temp_indices = identify_temperature_indices(array, key)
                
                # Store indices based on data type
                if 'obs' in key.lower():
                    obs_temp_indices = temp_indices
                    #print(f"  Stored observation temperature indices: {temp_indices}")
                elif 'act' in key.lower():
                    acts_temp_indices = temp_indices
                    print(f"  Stored action temperature indices: {temp_indices}")
                
                temperature_indices[key] = temp_indices
    
    # Second pass: Apply conversions using the identified indices
    print(f"\n=== Applying Temperature Conversions ===")
    
    for key in data.keys():
        array = data[key]
        print(f"\nConverting {key} (shape: {array.shape}):")
        
        if array.dtype == object:
            # Handle object arrays
            converted_array = array.copy()
            
            if array.ndim == 1:
                # Determine which temperature indices to use
                if 'obs' in key.lower() and obs_temp_indices is not None:
                    temp_indices = obs_temp_indices
                    #print(f"  Using observation temperature indices: {temp_indices}")
                elif 'act' in key.lower() and acts_temp_indices is not None:
                    temp_indices = acts_temp_indices
                    #print(f"  Using action temperature indices: {temp_indices}")
                else:
                    temp_indices = []
                    print(f"  No temperature indices found for {key}")
                
                # Convert all elements using the identified indices
                for i, element in enumerate(array):
                    if isinstance(element, (list, np.ndarray)):
                        element_array = np.array(element)
                        if element_array.ndim == 1:
                            converted_element = element_array.copy()
                            for temp_idx in temp_indices:
                                if temp_idx < len(converted_element):
                                    converted_element[temp_idx] = element_array[temp_idx] - 273.15
                            converted_array[i] = converted_element.tolist()
                        else:
                            converted_array[i] = element
                    else:
                        converted_array[i] = element
                
                if temp_indices:
                    print(f"  Converted {len(temp_indices)} temperature signals in {key}")
            
            converted_data[key] = converted_array
            
        else:
            # Regular numpy array
            converted_array = array.copy()
            
            if array.ndim == 2:
                # Determine which temperature indices to use
                if 'obs' in key.lower() and obs_temp_indices is not None:
                    temp_indices = obs_temp_indices
                    #print(f"  Using observation temperature indices: {temp_indices}")
                elif 'act' in key.lower() and acts_temp_indices is not None:
                    temp_indices = acts_temp_indices
                    #print(f"  Using action temperature indices: {temp_indices}")
                else:
                    temp_indices = []
                    print(f"  No temperature indices found for {key}")
                
                # Convert temperature signals at identified indices
                for temp_idx in temp_indices:
                    converted_array[:, temp_idx] = array[:, temp_idx] - 273.15
                
                if temp_indices:
                    print(f"  Converted {len(temp_indices)} temperature signals in {key}")
            
            else:
                # For other dimensions, just copy the array
                print(f"  Skipped {key} (not 2D array)")
            
            converted_data[key] = converted_array
    
    # Save the converted data
    np.savez(output_file, **converted_data)
    
    print(f"\n=== Conversion Summary ===")
    total_temp_signals = sum(len(indices) for indices in temperature_indices.values())
    print(f"Total temperature signals identified: {total_temp_signals}")
    
    """
    if obs_temp_indices:
        print(f"  Observation temperature indices: {obs_temp_indices}")
    if acts_temp_indices:
        print(f"  Action temperature indices: {acts_temp_indices}")
    """
    print(f"\nConverted file saved to: {output_file}")
    
    # Save conversion report
    report = {
        'input_file': input_file,
        'output_file': output_file,
        'conversion_timestamp': np.datetime64('now').astype(str),
        'temperature_indices': temperature_indices,
        'observation_temperature_indices': obs_temp_indices,
        'action_temperature_indices': acts_temp_indices,
        'total_signals_analyzed': len(data.keys()),
        'conversion_rule': 'Kelvin to Celsius (K - 273.15)',
        'identification_criteria': {
            'mean_range': '273-350K',
            'variance_threshold': '< 50',
            'description': 'Signals with mean > 273K and variance < 50 identified as temperatures'
        }
    }
    
    report_path = output_file.replace('.npz', '_temp_conversion_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Conversion report saved to: {report_path}")

def main():
    """
    Main function to analyze and merge expert trajectory files.
    """
    # File paths
    file1_path = "expert_trajectories_boptest_new_obs.npz"
    file2_path = "expert_trajectories_nocooling.npz"
    output_path = "expert_trajectories_merged.npz"
    
    # Temperature conversion paths
    boptest_converted_path = "expert_trajectories_boptest_celsius.npz"
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"expert_trajectory_processing_{timestamp}.log"
    tee = setup_logging(log_file_path)
    
    try:
        print("Expert Trajectory Analysis and Merging Tool")
        print("=" * 50)
        print(f"Logging output to: {log_file_path}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check if files exist
        if not os.path.exists(file1_path):
            print(f"Error: {file1_path} not found!")
            return
        
        if not os.path.exists(file2_path):
            print(f"Error: {file2_path} not found!")
            return
        
        # Step 1: Convert boptest temperatures from Kelvin to Celsius
        print("\nStep 1: Converting boptest temperatures from Kelvin to Celsius...")
        convert_boptest_temperatures(file1_path, boptest_converted_path)
        
        # Step 2: Perform the merge using the converted boptest file
        print("\nStep 2: Merging trajectory files...")
        merge_trajectory_files(boptest_converted_path, file2_path, output_path)
        
        print("\n=== Analysis Complete ===")
        print("The temperature conversion and merging are complete.")
        print(f"- Converted boptest file: {boptest_converted_path}")
        print(f"- Merged file: {output_path}")
        print("- Detailed reports have been saved.")
        
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All output has been saved to: {log_file_path}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()

if __name__ == "__main__":
    main()
