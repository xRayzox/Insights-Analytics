import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_league_table
)

# Retrieve data
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])

# Get the current game week
ct_gw = get_current_gw()

# Reset indices for DataFrames
sui = team_fixt_df.reset_index()
val = team_fdr_df.reset_index()

# Rename the first column to 'Team'
sui.rename(columns={0: 'Team'}, inplace=True)
val.rename(columns={0: 'Team'}, inplace=True)

# Combine FDR values from 'val' DataFrame
fdr_matrix = val.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Convert FDR values to integers
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(str)

# Create a filtered FDR matrix for styling
filtered_fdr_matrix = sui.copy()
filtered_fdr_matrix = filtered_fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='DisplayValue')

# Merge with FDR values
merged_fdr_matrix = pd.merge(filtered_fdr_matrix, fdr_matrix, on=['Team', 'GameWeek'], how='left')

# Define the custom color mapping for FDR values
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Define a coloring function based on the FDR values using the custom color mapping
def color_fdr(value):
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color};'
    else:
        return ''  # No style for undefined values

# Create a color mapping for each FDR value
color_mapping = merged_fdr_matrix.set_index(['Team', 'GameWeek'])['FDR'].map(color_fdr)

# Pivot to create the filtered matrix for display
styled_fdr_matrix = merged_fdr_matrix.pivot(index='Team', columns='GameWeek', values='DisplayValue')

# Rename columns for display purposes
styled_fdr_matrix.columns = [f'GW {col}' for col in styled_fdr_matrix.columns]

# Apply the styling using the new mapping
styled_filtered_fdr_table = styled_fdr_matrix.style.applymap(lambda x: color_mapping.loc[x.name])

# Streamlit app to display the styled table
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)