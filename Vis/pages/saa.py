import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
)

# Fetch data
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])

# Data processing
gw_min = min(events_df['id'])
gw_max = max(events_df['id'])
ct_gw = get_current_gw()

sui = team_fixt_df.reset_index()
val = team_fdr_df.reset_index()

# Rename the first column to 'Team'
sui.rename(columns={0: 'Team'}, inplace=True)
val.rename(columns={0: 'Team'}, inplace=True)

# Create new column names, keeping 'Team' as the first column
sui.columns = ['Team'] + [f'GW{col}' for col in range(1, len(sui.columns))]
val.columns = ['Team'] + [f'GW{col}' for col in range(1, len(val.columns))]

# Combine teams from both DataFrames
teams = pd.concat([sui['Team'], val['Team']]).unique()

# Initialize the FDR matrix
fdr_matrix = pd.DataFrame(index=teams, columns=sui.columns)

# Populate the FDR matrix based on GW matches using actual values
for team in teams:
    # Retrieve the FDR values for the current team
    fdr_values = val[val['Team'] == team].values.flatten()[1:]  # Exclude the 'Team' column
    fdr_matrix.loc[team] = fdr_values

# Convert FDR matrix to a format suitable for styling
fdr_matrix = fdr_matrix.reset_index()
fdr_matrix = fdr_matrix.melt(id_vars='index', var_name='GameWeek', value_name='FDR')
fdr_matrix.rename(columns={'index': 'Team'}, inplace=True)

# Define a coloring function based on the FDR values
def color_fdr(value):
    if value <= 2:
        return 'background-color: red'  # High difficulty
    elif value <= 4:
        return 'background-color: yellow'  # Medium difficulty
    else:
        return 'background-color: green'  # Low difficulty

# Create a filtered FDR matrix for styling
filtered_fdr_matrix = fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Apply the styling to the filtered FDR matrix
styled_filtered_fdr_table = filtered_fdr_matrix.style.applymap(color_fdr)

# Streamlit app to display the styled table
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)
