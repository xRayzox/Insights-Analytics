import streamlit as st
import pandas as pd
import glob
import os
import time


@st.cache_data  # Cache the combined DataFrame
def load_manager_data(data_path="./data/manager/clean_Managers_part*.csv"):
    start_time = time.time()
    all_files = glob.glob(data_path)  

    if not all_files:
        st.error(f"No files found at path: {data_path}")  # Handle empty directory
        return pd.DataFrame() # Return empty DataFrame if no files found.

    try:
        chunks = [pd.read_csv(file) for file in all_files]
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading data: {e}") # Display detailed error message
        return pd.DataFrame()

    end_time = time.time()
    st.info(f"Data loaded in {end_time - start_time:.2f} seconds")
    return df



df_managers = load_manager_data()

