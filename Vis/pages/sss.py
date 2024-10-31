import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
# Assuming fpl_api_collection and fpl_utils are in the FPL directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'FPL')))

from fpl_api_collection import (
    get_player_id_dict, get_bootstrap_data, get_player_data, get_league_table,
    get_fixt_dfs, get_current_gw, remove_moved_players, get_current_season
)
from fpl_utils import define_sidebar

st.set_page_config(page_title='Player Stats', page_icon=':shirt:', layout='wide')
define_sidebar()
st.title("Players")

###########################################################
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

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]

league_df = league_df.join(new_fixt_df)

float_cols = league_df.select_dtypes(include='float64').columns.values

league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1

league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])

## Very slow to load, works but needs to be sped up.
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')
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
    for i in range(1,6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict


home_away_dict = get_home_away_str_dict()


def color_fixtures(val):
    bg_color = 'background-color: '
    font_color = 'color: '
    if val in home_away_dict[1]:
        bg_color += '#147d1b'  # Green for easiest
    elif val in home_away_dict[2]:
        bg_color += '#00ff78'  # Lighter green
    elif val in home_away_dict[3]:
        bg_color += '#eceae6'  # Light gray for neutral
    elif val in home_away_dict[4]:
        bg_color += '#ff0057'  # Red for harder
        font_color += 'white'
    elif val in home_away_dict[5]:
        bg_color += '#920947'  # Darker red for hardest
        font_color += 'white'
    else:
        bg_color += ''
    style = bg_color + '; ' + font_color
    return style

def convert_score_to_result(df):
    df.loc[df['was_home'] == True, 'result'] = df['team_h_score'] \
        .astype('Int64').astype(str) \
        + '-' + df['team_a_score'].astype('Int64').astype(str)
    df.loc[df['was_home'] == False, 'result'] = df['team_a_score'] \
        .astype('Int64').astype(str) \
        + '-' + df['team_h_score'].astype('Int64').astype(str)
        
def convert_opponent_string(df):
    df.loc[df['was_home'] == True, 'vs'] = df['vs'] + ' (H)'
    df.loc[df['was_home'] == False, 'vs'] = df['vs'] + ' (A)'


def collate_hist_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    p_data = get_player_data(str(p_id[0]))
    p_df = pd.DataFrame(p_data['history'])
    convert_score_to_result(p_df)
    p_df.loc[p_df['result'] == '<NA>-<NA>', 'result'] = '-'
    rn_dict = {'round': 'GW', 'opponent_team': 'vs', 'total_points': 'Pts',
               'minutes': 'Mins', 'goals_scored': 'GS', 'assists': 'A',
               'clean_sheets': 'CS', 'goals_conceded': 'GC', 'own_goals': 'OG',
               'penalties_saved': 'Pen_Save', 'penalties_missed': 'Pen_Miss',
               'yellow_cards': 'YC', 'red_cards': 'RC', 'saves': 'S',
               'bonus': 'B', 'bps': 'BPS', 'influence': 'I', 'creativity': 'C',
               'threat': 'T', 'ict_index': 'ICT', 'value': 'Price',
               'selected': 'SB', 'transfers_in': 'Tran_In',
               'transfers_out': 'Tran_Out', 'expected_goals': 'xG',
               'expected_assists': 'xA', 'expected_goal_involvements': 'xGI',
               'expected_goals_conceded': 'xGC', 'result': 'Result'}
    p_df.rename(columns=rn_dict, inplace=True)
    col_order = ['GW', 'vs', 'Result', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA',
                 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S',
                 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB',
                 'Tran_In', 'Tran_Out', 'was_home']
    p_df = p_df[col_order]
    # map opponent teams
    p_df['Price'] = p_df['Price']/10
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    convert_opponent_string(p_df)
    p_df.drop('was_home', axis=1, inplace=True)
    p_df.set_index('GW', inplace=True)
    p_df.sort_values('GW', ascending=False, inplace=True)
    return p_df


def collate_total_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    df = ele_df.copy()
    p_total_df = df.loc[df['id'] == p_id[0]]
    p_total_df = p_total_df.copy()
    p_gw_df = collate_hist_df_from_name(player_name)
    p_total_df['GP'] = (p_gw_df['Mins'] > 0).sum()
    p_total_df['xG'] = p_gw_df['xG'].astype(float).sum()
    p_total_df['xA'] = p_gw_df['xA'].astype(float).sum()
    p_total_df['xGI'] = p_gw_df['xGI'].astype(float).sum()
    p_total_df['xGC'] = p_gw_df['xGC'].astype(float).sum()
    # index = web_name
    rn_dict = {'form': 'Form', 'points_per_game': 'PPG', 'total_points': 'Pts',
               'minutes': 'Mins', 'goals_scored': 'GS', 'clean_sheets': 'CS',
               'goals_conceded': 'GC', 'own_goals': 'OG', 'assists': 'A',
               'penalties_saved': 'Pen_Save', 'now_cost': 'Price',
               'penalties_missed': 'Pen_Miss', 'yellow_cards': 'YC',
               'red_cards': 'RC', 'saves': 'S', 'bonus': 'B', 'bps': 'BPS',
               'selected_by_percent': 'TSB%', 'influence': 'I',
               'creativity': 'C', 'threat': 'T', 'ict_index': 'ICT'}
    p_t = p_total_df.rename(columns=rn_dict)
    col_order = ['web_name', 'team', 'GP', 'PPG', 'Pts', 'Mins', 'GS',
                 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG',
                 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C',
                 'T', 'ICT', 'Form', 'TSB%', 'element_type']
    p_t = p_t[col_order]
    p_t['Price'] = p_t['Price']/10
    p_t['TSB%'] = p_t['TSB%'].replace('0.0', '0.09') 
    p_t['TSB%'] = p_t['TSB%'].astype(float)/100
    p_t.set_index('web_name', inplace=True)
    return p_t


def collated_spider_df_from_name(player_name):
    sp_df = collate_total_df_from_name(player_name)
    p_df = collate_hist_df_from_name(player_name)
    # league_df = get_league_table()
    # sp_df['gp'] = sp_df['team'].map(league_df.set_index('id')['GP'])
    sp_df['gp'] = len(p_df)
    sp_df['90s'] = sp_df['Mins']/90
    sp_df['G/90'] = sp_df['GS']/sp_df['90s']
    sp_df['xG/90'] = sp_df['xG']/sp_df['90s']
    sp_df['A/90'] = sp_df['A']/sp_df['90s']
    sp_df['xA/90'] = sp_df['xA']/sp_df['90s']
    sp_df['xGI/90'] = sp_df['xGI']/sp_df['90s']
    sp_df['BPS/90'] = sp_df['BPS']/sp_df['90s']
    sp_df['Ave_Mins'] = sp_df['Mins']/sp_df['gp']
    sp_df['Influence/90'] = sp_df['I'].astype(float)/sp_df['90s']
    sp_df['Creativity/90'] = sp_df['C'].astype(float)/sp_df['90s']
    sp_df['Threat/90'] = sp_df['T'].astype(float)/sp_df['90s']
    sp_df['ICT/90'] = sp_df['ICT'].astype(float)/sp_df['90s']
    sp_df['CS/90'] = sp_df['CS']/sp_df['90s']
    sp_df['GC/90'] = sp_df['GC']/sp_df['90s']
    sp_df['xGC/90'] = sp_df['xGC']/sp_df['90s']
    sp_df['YC/90'] = sp_df['YC']/sp_df['90s']
    sp_df['B/90'] = sp_df['B']/sp_df['90s']
    sp_df['S/90'] = sp_df['S']/sp_df['90s']
    return sp_df

##########################################################################
def display_frame(df):
    '''display dataframe with all float columns rounded to 1 decimal place'''
    float_cols = df.select_dtypes(include='float64').columns.values
    st.dataframe(df.style.format(subset=float_cols, formatter='{:.1f}'))


def get_ICT_spider_plot(player_name1, player_name2):
    cats = ['BPS/90', 'Ave_Mins', 'Influence/90', 'Creativity/90', 'Threat/90',
            'ICT/90']
    sp1_df = collated_spider_df_from_name(player_name1)
    sp1_df['player_name'] = player_name1
    sp1_df.set_index('player_name', inplace=True)
    sp1_df = sp1_df[cats].transpose().reset_index()
    
    sp2_df = collated_spider_df_from_name(player_name2)
    sp2_df['player_name'] = player_name2
    sp2_df.set_index('player_name', inplace=True)
    sp2_df = sp2_df[cats].transpose().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(name=player_name1, r=list(sp1_df[player_name1]), theta=list(sp1_df['index'])))
    fig.add_trace(go.Scatterpolar(name=player_name2, r=list(sp2_df[player_name2]), theta=list(sp2_df['index'])))
    fig.update_layout(legend=dict(x=0.33, y=1.25))
    return fig


def get_stats_spider_plot(player_name1, player_name2):
    cats = ['G/90', 'xG/90', 'A/90', 'xA/90', 'xGI/90', 'CS/90', 'GC/90',
            'xGC/90', 'YC/90', 'B/90', 'S/90']
    sp1_df = collated_spider_df_from_name(player_name1)
    sp1_df['player_name'] = player_name1
    sp1_df.set_index('player_name', inplace=True)
    sp1_df = sp1_df[cats].transpose().reset_index()
    
    sp2_df = collated_spider_df_from_name(player_name2)
    sp2_df['player_name'] = player_name2
    sp2_df.set_index('player_name', inplace=True)
    sp2_df = sp2_df[cats].transpose().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(name=player_name1, r=list(sp1_df[player_name1]), theta=list(sp1_df['index'])))
    fig.add_trace(go.Scatterpolar(name=player_name2, r=list(sp2_df[player_name2]), theta=list(sp2_df['index'])))
    fig.update_layout(legend=dict(x=0.33, y=1.25))
    return fig


def get_top_two_mid_ids():
    ele_data = get_bootstrap_data()['elements']
    ele_df = pd.DataFrame(ele_data)
    mid_df = ele_df.loc[ele_df['element_type'] == 3]
    mid_df_sorted = mid_df.sort_values('total_points', ascending=False)
    top2 = mid_df_sorted.head(2).reset_index()
    return top2['id'][0], top2['id'][1]


def get_player_next3(player):
    player_team = player[len(player) - 4:].replace(')', '')
    player_next3 = league_df.loc[league_df['Team'] == player_team][new_fixt_cols]
    player_next3['Team'] = player_team + ' next 3:'
    player_next3.set_index('Team', inplace=True)
    return player_next3
    
#######################################################################
price_min = (ele_copy['now_cost'].min())/10
price_max = (ele_copy['now_cost'].max())/10
price_min = float(price_min)
price_max = float(price_max)
if len(get_player_data(list(full_player_dict.keys())[0])['history']) == 0:
    st.write(f"Please wait for the {crnt_season} season to begin for individual player statistics")
else:
    filter_rows = st.columns([2,3])
    filter_pos = filter_rows[0].multiselect(
        'Filter Position',
        ['GKP', 'DEF', 'MID', 'FWD'],
        ['GKP', 'DEF', 'MID', 'FWD']
    )
    slider1, slider2 = filter_rows[1].slider('Filter Price: ', price_min, price_max, [price_min, price_max], 0.1, format='£%.1f')
###
    ele_copy['team_name'] = ele_copy['team'].map(teams_df.set_index('id')['short_name'])
    ele_copy['price'] = ele_copy['now_cost']/10
    ele_copy = remove_moved_players(ele_copy)
    ele_cut = ele_copy.loc[(ele_copy['price'] <= slider2) &
                            (ele_copy['price'] > slider1) &
                            (ele_copy['element_type'].isin(filter_pos))]
    
    ele_cut.sort_values('price', ascending=False, inplace=True)
    ele_cut['full_name'] = ele_cut['first_name'] + ' ' + \
        ele_cut['second_name'] + ' (' + ele_cut['team_name'] + ')'
    id_dict = dict(zip(ele_cut['id'], ele_cut['full_name']))
    ele_cut['logo_player'] = "https://resources.premierleague.com/premierleague/photos/players/250x250/p" + ele_cut['code'].astype(str) + ".png"


    if len(id_dict) == 0:
        st.write('No data to display in range.')
    elif len(id_dict) >= 1:
        init_rows = st.columns(4)
        player1 = init_rows[0].selectbox("Choose Player One", id_dict.values(), index=0)
        player1_next3 = get_player_next3(player1)
        for col in new_fixt_cols:
            if player1_next3[col].dtype == 'O':
                max_length = player1_next3[col].str.len().max()
                if max_length > 7:
                    # Correct use of .loc for both selection and assignment:
                    player1_next3.loc[player1_next3[col].str.len() <= 7, col] = player1_next3.loc[player1_next3[col].str.len() <= 7, col].str.pad(width=max_length+9, side='both', fillchar=' ')

        styled_player1_next3 = player1_next3.style.map(color_fixtures, subset=new_fixt_df.columns) \
                .format(subset=player1_next3.select_dtypes(include='float64') \
                        .columns.values, formatter='{:.2f}')
        init_rows[1].dataframe(styled_player1_next3)
            
        if len(id_dict) > 1:
            player2 = init_rows[2].selectbox("Choose Player Two", id_dict.values(), 1) #index=int(ind2))
            player2_next3 = get_player_next3(player2)
            for col in new_fixt_cols:
                if player2_next3[col].dtype == 'O':
                    max_length = player2_next3[col].str.len().max()
                    if max_length > 7:
                        # Correct use of .loc for selection and assignment:
                        player2_next3.loc[player2_next3[col].str.len() <= 7, col] = player2_next3.loc[player2_next3[col].str.len() <= 7, col].str.pad(width=max_length+9, side='both', fillchar=' ')
            
            styled_player2_next3 = player2_next3.style.map(color_fixtures, subset=new_fixt_df.columns) \
                    .format(subset=player2_next3.select_dtypes(include='float64') \
                            .columns.values, formatter='{:.2f}')
            init_rows[3].dataframe(styled_player2_next3)
            
        rows = st.columns(2)
        player1_df = collate_hist_df_from_name(player1)
        player1_total_df = collate_total_df_from_name(player1)
        player1_total_df.drop(['team', 'element_type'], axis=1, inplace=True)
        #rows[0].dataframe(player1_df.style.format({'Price': '£{:.1f}'}))
        total_fmt = {'xG':'{:.2f}', 'xA':'{:.2f}', 'xGI':'{:.2f}', 'xGC':'{:.2f}',
                     'Price': '£{:.1f}', 'TSB%': '{:.1%}'}
        rows[0].dataframe(player1_total_df.style.format(total_fmt))
        
        if len(id_dict) > 1:
            player2_df = collate_hist_df_from_name(player2)
            player2_total_df = collate_total_df_from_name(player2)
            player2_total_df.drop(['team', 'element_type'], axis=1, inplace=True)
            rows[1].dataframe(player2_df.style.format({'Price': '£{:.1f}'}))
            total_fmt = {'xG':'{:.2f}', 'xA':'{:.2f}', 'xGI':'{:.2f}', 'xGC':'{:.2f}',
                     'Price': '£{:.1f}', 'TSB%': '{:.1%}'}
            rows[1].dataframe(player2_total_df.style.format(total_fmt))

            rows[0].plotly_chart(get_stats_spider_plot(player1, player2))
            rows[1].plotly_chart(get_ICT_spider_plot(player1, player2))
        

#st.plotly_chart(get_spider_plot(player1, player2), use_container_width=True)correct this
 