import polars as pl
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# Directory containing the CSV files
converted_path = './data/manager/'

# List all CSV files
csv_files = glob.glob(os.path.join(converted_path, '*.csv'))

# Function to load a single CSV file
def load_csv(file):
    try:
        # Polars' CSV reader with optimized options
        df = pl.read_csv(file)
        print(f"Loaded {os.path.basename(file)}: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {os.path.basename(file)}: {e}")
        return None  # Handle errors gracefully

# Measure loading time with parallelism
def measure_parallel_loading():
    print("Parallel CSV Loading with Polars...")
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor() as executor:
        # Filter out any None (failed loads)
        dataframes = [df for df in executor.map(load_csv, csv_files) if df is not None]

    total_time = time.time() - start_time
    print(f"Total time for parallel loading: {total_time:.2f} seconds\n")
    return dataframes

# Load all dataframes in parallel
dfs_parallel = measure_parallel_loading()

# Combine the dataframes efficiently
if dfs_parallel:  # Ensure there are valid dataframes
    combined_df = pl.concat(dfs_parallel, how='vertical')

    # Filter rows where the 'Manager' column contains "Wael Hc"
    if "Manager" in combined_df.columns:
        filtered_df = combined_df.filter(combined_df['Manager'].str.contains("Wael"))
        print(filtered_df)
    else:
        print("'Manager' column not found in the loaded data.")
else:
    print("No dataframes were loaded successfully.")


st.write(filtered_df)