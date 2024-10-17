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
    get_fixt_dfs
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

# Sort val DataFrame based on the first game week (for example)
val_sorted = val.sort_values(by='GW1')

# Create a DataFrame with the sorted order of teams from val
sorted_teams = val_sorted['Team'].values

# Reorder sui based on the sorted teams from val
sui_sorted = sui[sui['Team'].isin(sorted_teams)].set_index('Team').reindex(sorted_teams)

# Display the sorted sui DataFrame
st.title("Sorted Fixture Information")
st.dataframe(sui_sorted.reset_index())
