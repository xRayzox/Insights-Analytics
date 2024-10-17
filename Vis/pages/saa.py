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
from fpl_utils import (
    define_sidebar,
    get_annot_size,
    map_float_to_color,
    get_text_color_from_hash,
    get_rotation
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

# Create FDR matrix directly from 'val' DataFrame
fdr_matrix = val.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Convert FDR values to integers
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)

slider1, slider2 = st.slider('Gameweek: ', gw_min, gw_max, [int(ct_gw), int(ct_gw+4)], 1)
annot_size = get_annot_size(slider1, slider2)
rotation = get_rotation(slider1, slider2)
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
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    else:
        return ''  # No style for undefined values

# Create a filtered FDR matrix for styling
filtered_fdr_matrix = fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Rename columns for display purposes
filtered_fdr_matrix.columns = [f'GW {col}' for col in filtered_fdr_matrix.columns]

# Apply the styling to the filtered FDR matrix
styled_filtered_fdr_table = filtered_fdr_matrix.style.applymap(color_fdr)

# Streamlit app to display the styled table
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)