import streamlit as st
import pandas as pd
import glob
import os
import time
import dask.dataframe as dd  # Use Dask for parallel processing

@st.cache_data  # Cache the combined DataFrame
def load_manager_data_optimized(data_path="./data/manager/clean_Managers_part*.csv"):
    start_time = time.time()
    all_files = glob.glob(data_path)  

    if not all_files:
        st.error(f"No files found at path: {data_path}")  
        return pd.DataFrame()

    #Infer dtypes from the first file.
    dtypes = pd.read_csv(all_files[0], nrows=0).dtypes.to_dict()


    try:
        # Use Dask to read CSV files in parallel with specified data types
        df = dd.read_csv(all_files, dtypes=dtypes).compute()

        # Convert string columns with repeated values to 'category' dtype
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
             if df[col].nunique() / len(df) < 0.1: # Threshold for low cardinality
                df[col] = df[col].astype('category')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


    end_time = time.time()
    st.info(f"Data loaded in {end_time - start_time:.2f} seconds.  Shape: {df.shape}")
    return df


# Load the data using the optimized function
df_managers = load_manager_data_optimized()


