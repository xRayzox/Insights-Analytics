import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
import datetime as datetime
import os
import sys

import numpy as np
import plotly.express as px

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

# Populate the FDR matrix based on GW matches
for team in teams:
    # Calculate FDR values based on team matchups
    gw1_fdr = np.random.randint(1, 6)  # Example random score for GW1
    gw2_fdr = np.random.randint(1, 6)  # Example random score for GW2
    fdr_matrix.loc[team] = [gw1_fdr, gw2_fdr]

# Convert FDR matrix to a format suitable for styling
fdr_matrix = fdr_matrix.reset_index()
fdr_matrix = fdr_matrix.melt(id_vars='index', var_name='GameWeek', value_name='FDR')
fdr_matrix.rename(columns={'index': 'Team'}, inplace=True)

# Define a coloring function based on the FDR values
def color_fdr(team, game_week):
    value = fdr_matrix[(fdr_matrix['Team'] == team) & (fdr_matrix['GameWeek'] == game_week)]['FDR'].values[0]
    if value <= 2:
        return 'background-color: red'  # High difficulty
    elif value <= 4:
        return 'background-color: yellow'  # Medium difficulty
    else:
        return 'background-color: green'  # Low difficulty

# Create a filtered FDR matrix for styling
filtered_fdr_matrix = fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Apply the styling to the filtered FDR matrix
styled_filtered_fdr_table = filtered_fdr_matrix.style.apply(
    lambda row: [color_fdr(row.name, col) for col in row.index], axis=1
)

# Streamlit app to display the styled table
st.title("Fixture Difficulty Rating (FDR) Matrix")
st.write(styled_filtered_fdr_table)