import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from concurrent.futures import ThreadPoolExecutor ,ProcessPoolExecutor,as_completed
cwd = os.getcwd()
# Construct the full path to the 'FPL' directory
fpl_path = os.path.join(cwd, '..', '..', 'FPL')
# Add it to the system path
sys.path.append(fpl_path)
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_fixture_data,
    get_player_id_dict,
    get_current_season,
    get_player_data,
)


with open('./data/wave.css') as f:
        css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Retrieve and prepare player data
ele_types_data = get_bootstrap_data()['element_types']
ele_types_X_weighted = pd.DataFrame(ele_types_data)
ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)
ele_df['element_type'] = ele_df['element_type'].map(ele_types_X_weighted.set_index('id')['singular_name_short'])
ele_copy = ele_df.copy()

# Retrieve and prepare team data
teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)

# Map team IDs to names for fixture processing
team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()
ele_copy['team_name'] = ele_copy['team'].map(teams_df.set_index('id')['short_name'])
ele_copy['full_name'] = ele_copy['first_name'].str.cat(ele_copy['second_name'].str.cat(ele_copy['team_name'].apply(lambda x: f" ({x})"), sep=''), sep=' ')

# Retrieve player dictionary and current season/gameweek
full_player_dict = get_player_id_dict('total_points', web_name=False)
crnt_season = get_current_season()
ct_gw = get_current_gw()

# Retrieve and process fixture data
fixture_data = get_fixture_data()
fixtures_df = pd.DataFrame(fixture_data)
fixtures_df.drop(columns='stats', inplace=True)

fixtures_df['team_h'] = fixtures_df['team_h'].replace(team_name_mapping)
fixtures_df['team_a'] = fixtures_df['team_a'].replace(team_name_mapping)
fixtures_df = fixtures_df.drop(columns=['pulse_id'])

# Format fixture dates
timezone = 'Europe/London'
fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
fixtures_df['local_time'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%A %d %B %Y %H:%M')
fixtures_df['local_date'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%d %A %B %Y')
fixtures_df['local_hour'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%H:%M')

# Retrieve fixture difficulty rating data
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
full_player_dict = get_player_id_dict('total_points', web_name=False)