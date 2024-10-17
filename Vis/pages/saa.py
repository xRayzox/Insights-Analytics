import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
import datetime as datetime
import os
import sys

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
from fpl_params import (
    TIMEZONES_BY_CONTINENT,
    AUTHOR_CONTINENT,
    AUTHOR_CITY
)

team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])

gw_min = min(events_df['id'])
gw_max = max(events_df['id'])

ct_gw = get_current_gw()
sui = team_fixt_df.reset_index()
val = team_fdr_df.reset_index()

# Prepare for FDR Matrix Calculation
teams = sui[0].astype(str)  # Ensure teams are of type string
num_gameweeks = sui.shape[1] - 1  # Exclude team column
fdr_matrix = pd.DataFrame(index=teams, columns=[f'GW{i}' for i in range(num_gameweeks)])  # Initialize FDR matrix

# Populate FDR Matrix
for index, row in sui.iterrows():
    for gw in range(1, num_gameweeks + 1):
        fixture = row[gw]
        team_a = row[0]  # Home/Away team
        team_h = fixture.split()[0]  # Extracting opponent team name from 'TEAM (H/A)'

        # Retrieve the FDR value from the 'val' DataFrame
        fdr_value = val.loc[team_h, gw - 1] if team_h in val.index else None  # Adjust index to match gameweek

        # Store the FDR value in the matrix
        fdr_matrix.at[team_a, f'GW{gw - 1}'] = fdr_value

# Color Coding Function
def color_fdr(value):
    colors = {
        1: ('#257d5a', 'black'),
        2: ('#00ff86', 'black'),
        3: ('#ebebe4', 'black'),
        4: ('#ff005a', 'white'),
        5: ('#861d46', 'white'),
    }
    bg_color, font_color = colors.get(value, ('white', 'black'))
    return f'background-color: {bg_color}; color: {font_color};'

# Streamlit Display
st.title("Fixture Difficulty Rating (FDR) Matrix")
selected_gameweek = st.sidebar.slider(
    "Select Gameweek:",
    min_value=1,
    max_value=num_gameweeks,
    value=8
)

# Filter FDR Matrix for the selected Gameweek and the next 9 Gameweeks
filtered_fdr_matrix = fdr_matrix.loc[:, f'GW{selected_gameweek}':f'GW{selected_gameweek + 9}'].fillna('')

# Apply Color Coding with map
styled_fdr_table = filtered_fdr_matrix.style.map(color_fdr)  # Use map instead of applymap

# Display FDR Matrix
st.write(f"**Fixture Difficulty Rating (FDR) for Gameweek {selected_gameweek + 1} onwards:**")
st.dataframe(styled_fdr_table)

# Define Colors for the Legend
colors = {
    1: ('#257d5a', 'black'),
    2: ('#00ff86', 'black'),
    3: ('#ebebe4', 'black'),
    4: ('#ff005a', 'white'),
    5: ('#861d46', 'white'),
}

# FDR Legend
st.sidebar.markdown("**Legend:**")
for fdr, (bg_color, font_color) in colors.items():
    st.sidebar.markdown(
        f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
        f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
        f"</span>",
        unsafe_allow_html=True,
    )
