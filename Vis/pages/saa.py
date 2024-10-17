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

team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])

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

# Create FDR matrix directly from 'val' DataFrame
fdr_matrix = val.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Define a coloring function based on the FDR values
def color_fdr(value):
    if value <= 2:
        return 'background-color: green'  # Low difficulty
    elif value <= 4:
        return 'background-color: yellow'  # Medium difficulty
    else:
        return 'background-color: red'  # High difficulty

# Create a filtered FDR matrix for styling
filtered_fdr_matrix = fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Apply the styling to the filtered FDR matrix
styled_filtered_fdr_table = filtered_fdr_matrix.style.applymap(color_fdr)

# Streamlit app to display the styled table
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)