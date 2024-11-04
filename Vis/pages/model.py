import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Set pandas option to avoid silent downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

# Adjust path for FPL API collection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_fixture_data,
    get_player_id_dict,
    get_current_season
)

# Retrieve and prepare player data
ele_types_data = get_bootstrap_data()['element_types']
ele_types_df = pd.DataFrame(ele_types_data)
ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)
ele_df['element_type'] = ele_df['element_type'].map(ele_types_df.set_index('id')['singular_name_short'])
ele_df['logo_player'] = "https://resources.premierleague.com/premierleague/photos/players/250x250/p" + ele_df['code'].astype(str) + ".png"
ele_copy = ele_df.copy()

# Retrieve and prepare team data
teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"

# Map team IDs to names for fixture processing
team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()

# Retrieve player dictionary and current season/gameweek
full_player_dict = get_player_id_dict('total_points', web_name=False)
crnt_season = get_current_season()
ct_gw = get_current_gw()

# Retrieve and process fixture data
fixture_data = get_fixture_data()
fixtures_df = pd.DataFrame(fixture_data)
fixtures_df.drop(columns='stats', inplace=True)
fixtures_df = fixtures_df.merge(teams_df[['id', 'logo_url']], left_on='team_h', right_on='id', how='left').rename(columns={'logo_url': 'team_h_logo'})
fixtures_df = fixtures_df.merge(teams_df[['id', 'logo_url']], left_on='team_a', right_on='id', how='left').rename(columns={'logo_url': 'team_a_logo'})
fixtures_df['team_h'] = fixtures_df['team_h'].replace(team_name_mapping)
fixtures_df['team_a'] = fixtures_df['team_a'].replace(team_name_mapping)
fixtures_df = fixtures_df.drop(columns=['pulse_id'])

# Format fixture dates
timezone = st.session_state['timezone']
fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
fixtures_df['local_time'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%A %d %B %Y %H:%M')
fixtures_df['local_date'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%d %A %B %Y')
fixtures_df['local_hour'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%H:%M')

# Retrieve fixture difficulty rating data
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()


st.write(fixtures_df)
st.markdown("suiii")
st.write(ele_copy)
st.markdown("suiii")
st.write(teams_df)