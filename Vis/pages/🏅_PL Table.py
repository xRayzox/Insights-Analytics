import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image

# Append the path for your FPL API collection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

# Import functions from your FPL API
from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)
from fpl_utils import (
    define_sidebar
)

# Set up the Streamlit page
st.set_page_config(page_title='PL Table', page_icon=':sports-medal:', layout='wide')
define_sidebar()
st.title('Premier League Table')

# Get the league table and fixture data
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()

# Prepare new fixture DataFrame for upcoming game weeks
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df)

# Format the league DataFrame
league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)

# Fetch bootstrap data for teams
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"

# Create a mapping of team logos
team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

# Display logos and team names
for team_name in teams_df.short_name:
    if team_name in team_logo_mapping:
        st.image(team_logo_mapping[team_name], caption=team_name)

# Function to get home/away fixture strings
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length + 9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict

    merged_dict = {k: [] for k in range(1, 6)}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            merged_dict[key].extend(value)
    for k in merged_dict.keys():
        merged_dict[k] = list(set(merged_dict[k]))  # Remove duplicates
    return merged_dict

home_away_dict = get_home_away_str_dict()

# Function to style the fixtures
def color_fixtures(val):
    bg_color = 'background-color: '
    font_color = 'color: '
    if val in home_away_dict[1]:
        bg_color += '#147d1b'
    elif val in home_away_dict[1.5]:
        bg_color += '#0ABE4A'
    elif val in home_away_dict[2]:
        bg_color += '#00ff78'
    elif val in home_away_dict[2.5]:
        bg_color += "#caf4bd"
    elif val in home_away_dict[3]:
        bg_color += '#eceae6'
    elif val in home_away_dict[3.5]:
        bg_color += "#fa8072"
    elif val in home_away_dict[4]:
        bg_color += '#ff0057'
        font_color += 'white'
    elif val in home_away_dict[4.5]:
        bg_color += '#C9054F'
        font_color += 'white'
    elif val in home_away_dict[5]:
        bg_color += '#920947'
        font_color += 'white'
    
    return f"{bg_color}; {font_color}"

# Adjust column padding for display
for col in new_fixt_cols:
    if league_df[col].dtype == 'O':
        max_length = league_df[col].str.len().max()
        if max_length > 7:
            league_df.loc[league_df[col].str.len() <= 7, col] = league_df[col].str.pad(width=max_length + 9, side='both', fillchar=' ')

# Style the DataFrame
styled_df = league_df.style.applymap(color_fixtures, subset=new_fixt_cols) \
                            .format(subset=float_cols, formatter='{:.2f}')

# Display the styled DataFrame in Streamlit
st.dataframe(styled_df, height=740, use_container_width=True)
