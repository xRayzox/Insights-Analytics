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


#Efficient display with Streamlit AgGrid
import streamlit_aggrid as st_aggrid

if not df_managers.empty:  # Check if DataFrame is empty before proceeding
    gb = st_aggrid.GridOptionsBuilder.from_dataframe(df_managers)
    gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination with automatic page size
    gb.configure_grid_options(domLayout='normal') # Default grid layout
    grid_response = st_aggrid.AgGrid(
        df_managers,
        gridOptions=gb.build(),
        height=400, # Set height of grid
        width='100%' # Set width to 100% to occupy the available space
    )


#Optional: Allow downloading the full combined data:
    st.download_button("Download Full Data (CSV)", df_managers.to_csv(index=False).encode('utf-8'), "combined_managers.csv", "text/csv", key='download-csv')