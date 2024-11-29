import polars as pl
import glob
import os
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
        return df
    except Exception as e:
        print(f"Error loading {os.path.basename(file)}: {e}")
        return None  # Handle errors gracefully

# Cache manager names and IDs to avoid repeated computation
@st.cache_data
def load_manager_data():
    print("Loading CSV files in parallel...")
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor() as executor:
        dataframes = [df for df in executor.map(load_csv, csv_files) if df is not None]
    
    # Combine the dataframes efficiently
    if dataframes:
        # Concatenate dataframes and cache the result for future use
        combined_df = pl.concat(dataframes, how='vertical')
        
        # Extract manager names and IDs
        manager_data = combined_df.select(['Manager', 'ID']).unique()
        return manager_data
    return None

# Load manager data once and cache it
manager_data = load_manager_data()

if manager_data is not None:
    with st.container():
        # Input field for searching the FPL Manager
        fpl_id1_search = st.text_input('Search your FPL Manager:')

        # Filtering the manager names based on the search input
        if fpl_id1_search:
            filtered_managers = manager_data.filter(pl.col("Manager").str.to_lowercase().str.contains(fpl_id1_search.lower()))
        else:
            filtered_managers = manager_data

        # Extract the unique manager names for the selectbox
        manager_names = filtered_managers['Manager'].to_list()

        # Providing the selectbox for the filtered Manager names
        selected_manager_name = st.selectbox('Please select your FPL Manager:', manager_names)

        # Get the corresponding Manager ID after the selection
        selected_manager_id = filtered_managers.filter(pl.col("Manager") == selected_manager_name)['ID'].to_list()

        # Display selected Manager ID
        if selected_manager_id:
            st.write(f"Selected Manager ID: {selected_manager_id[0]}")
else:
    st.error("Failed to load manager data.")
