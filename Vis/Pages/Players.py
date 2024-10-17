import streamlit as st
import pandas as pd
import sys
import os

# Adjust the system path to include the FPL directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

from fpl_api_collection import (
    get_player_id_dict, get_bootstrap_data, get_player_data, get_league_table,
    get_fixt_dfs, get_current_gw, remove_moved_players, get_current_season
)
import plotly.graph_objects as go
from fpl_utils import (
    define_sidebar
)

st.set_page_config(page_title='Player Stats', page_icon=':shirt:', layout='wide')
define_sidebar()

st.title("Players")
st.write("Currently only looking at data available through the FPL API. FBRef and Understat data being added is on the to-do list.")

# Load data from the FPL API
ele_types_data = get_bootstrap_data()['element_types']
ele_types_df = pd.DataFrame(ele_types_data)
ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)
ele_df['element_type'] = ele_df['element_type'].map(ele_types_df.set_index('id')['singular_name_short'])
ele_copy = ele_df.copy()

teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)

full_player_dict = get_player_id_dict('total_points', web_name=False)
crnt_season = get_current_season()

# Define columns of interest
ele_cols = ['id', 'web_name', 'chance_of_playing_this_round', 'element_type',
            'event_points', 'form', 'now_cost', 'points_per_game',
            'selected_by_percent', 'team', 'total_points',
            'transfers_in_event', 'transfers_out_event', 'value_form',
            'value_season', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
            'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'influence_rank', 'influence_rank_type', 'creativity_rank',
            'creativity_rank_type', 'threat_rank', 'threat_rank_type',
            'ict_index_rank', 'ict_index_rank_type', 'dreamteam_count']

ele_df = ele_df[ele_cols]
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()

# Prepare fixture DataFrame
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df)

# Format goal difference
league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])

# Get home and away fixtures as a dictionary
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
    
    merged_dict = {}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        decoupled_list = list(set(v))
        merged_dict[k] = decoupled_list
    for i in range(1, 6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict

home_away_dict = get_home_away_str_dict()

# Function to apply color styling to fixtures
def color_fixtures(val):
    bg_color = 'background-color: '
    font_color = 'color: '
    if val in home_away_dict[1]:
        bg_color += '#147d1b'
    elif val in home_away_dict[2]:
        bg_color += '#00ff78'
    elif val in home_away_dict[3]:
        bg_color += '#eceae6'
    elif val in home_away_dict[4]:
        bg_color += '#ff0057'
        font_color += 'white'
    elif val in home_away_dict[5]:
        bg_color += '#920947'
        font_color += 'white'
    else:
        bg_color += ''
    style = bg_color + '; ' + font_color
    return style

# Functions for player data transformation
def convert_score_to_result(df):
    df.loc[df['was_home'] == True, 'result'] = df['team_h_score'].astype('Int64').astype(str) + '-' + df['team_a_score'].astype('Int64').astype(str)
    df.loc[df['was_home'] == False, 'result'] = df['team_a_score'].astype('Int64').astype(str) + '-' + df['team_h_score'].astype('Int64').astype(str)

def convert_opponent_string(df):
    df.loc[df['was_home'] == True, 'vs'] = df['vs'] + ' (H)'
    df.loc[df['was_home'] == False, 'vs'] = df['vs'] + ' (A)'

# Function to collate historical data for a player
def collate_hist_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    p_data = get_player_data(str(p_id[0]))
    p_df = pd.DataFrame(p_data['history'])
    convert_score_to_result(p_df)
    p_df.loc[p_df['result'] == '<NA>-<NA>', 'result'] = '-'
    rn_dict = {
        'round': 'GW', 'opponent_team': 'vs', 'total_points': 'Pts',
        'minutes': 'Mins', 'goals_scored': 'GS', 'assists': 'A',
        'clean_sheets': 'CS', 'goals_conceded': 'GC', 'own_goals': 'OG',
        'penalties_saved': 'Pen_Save', 'penalties_missed': 'Pen_Miss',
        'yellow_cards': 'YC', 'red_cards': 'RC', 'saves': 'S',
        'bonus': 'B', 'bps': 'BPS', 'influence': 'I', 'creativity': 'C',
        'threat': 'T', 'ict_index': 'ICT', 'value': 'Price',
        'selected': 'SB', 'transfers_in': 'Tran_In',
        'transfers_out': 'Tran_Out', 'expected_goals': 'xG',
        'expected_assists': 'xA', 'expected_goal_involvements': 'xGI',
        'expected_goals_conceded': 'xGC', 'result': 'Result'
    }
    p_df.rename(columns=rn_dict, inplace=True)
    col_order = ['GW', 'vs', 'Result', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA',
                 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S',
                 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB',
                 'Tran_In', 'Tran_Out', 'was_home']
    p_df = p_df[col_order]
    p_df['Price'] = p_df['Price'] / 10
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    convert_opponent_string(p_df)
    p_df.drop('was_home', axis=1, inplace=True)
    p_df.set_index('GW', inplace=True)
    p_df.sort_values('GW', ascending=False, inplace=True)
    return p_df

# Function to collate total player statistics
def collate_total_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    df = ele_df.copy()
    p_total_df = df.loc[df['id'] == p_id[0]]
    p_total_df = p_total_df.copy()
    p_gw_df = collate_hist_df_from_name(player_name)
    return p_total_df, p_gw_df

# Main Streamlit App Logic
col1, col2, col3 = st.columns([3, 3, 1])

with col1:
    selected_positions = st.multiselect('Select Positions', options=ele_df['element_type'].unique(), default=ele_df['element_type'].unique().tolist())
    selected_team = st.multiselect('Select Team', options=teams_df['name'].unique(), default=teams_df['name'].unique().tolist())
    selected_players = st.multiselect('Select Players', options=ele_df['web_name'].unique())

with col2:
    if selected_players:
        id_dict = {ele_df.loc[ele_df['web_name'] == player, 'id'].values[0]: player for player in selected_players}
        st.write(f"Selected Players: {', '.join(id_dict.values())}")
    else:
        id_dict = {}

# Handle filtering based on selected positions and teams
if selected_positions or selected_team:
    filtered_players = ele_df
    if selected_positions:
        filtered_players = filtered_players[filtered_players['element_type'].isin(selected_positions)]
    if selected_team:
        filtered_players = filtered_players[filtered_players['team'].isin(selected_team)]

    # Display filtered player data
    if not filtered_players.empty:
        st.write(filtered_players)
    else:
        st.write('No players found matching the selected criteria.')

else:
    st.write('Please select a position or team to filter players.')

if len(id_dict) == 0:
    st.write('No players found matching the selected criteria.')
else:
    selected_players = st.selectbox("Select Players to Compare", options=list(id_dict.values()))
    
    if st.button('Show Player Stats'):
        # Display selected player's stats
        player_stats_df, player_gw_df = collate_total_df_from_name(selected_players)
        st.subheader(f"Total Stats for {selected_players}:")
        st.write(player_stats_df)

        # Get the next 3 fixtures for the selected player
        player_next_fixtures = get_player_next3(selected_players)
        st.subheader(f"Next 3 Fixtures for {selected_players}:")
        st.write(player_next_fixtures)
