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

# Combine teams from both DataFrames
teams = pd.concat([sui['Team'], val['Team']]).unique()

# Melt both DataFrames for easier manipulation
sui_melted = sui.melt(id_vars='Team', var_name='GameWeek', value_name='Fixture')
val_melted = val.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Merge the melted DataFrames on 'Team' and 'GameWeek'
fdr_matrix = pd.merge(sui_melted, val_melted, on=['Team', 'GameWeek'])

# Convert FDR values to integers
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)

# Define the custom color mapping for FDR values
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Define a coloring function based on the FDR values
def color_fdr(row):
    value = row['FDR']
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    else:
        return ''  # No style for undefined values

# Pivot the fdr_matrix for display
filtered_fdr_matrix = fdr_matrix.pivot(index='Team', columns='GameWeek', values='Fixture')
filtered_fdr_matrix.columns = [f'GW {col}' for col in filtered_fdr_matrix.columns]

# Apply the styling
styled_filtered_fdr_table = filtered_fdr_matrix.style.apply(color_fdr, axis=1)

# Streamlit app
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)