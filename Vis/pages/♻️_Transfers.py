import streamlit as st
import pandas as pd
import altair as alt
import sys
import os
import subprocess
import threading
import schedule
import time
import sys
import subprocess
pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'FPL')))



from fpl_api_collection import (
    get_bootstrap_data, get_total_fpl_players, get_player_id_dict, get_player_data
)
from fpl_utils import (
    define_sidebar
)
st.set_page_config(page_title='Transfers', page_icon=':recycle:', layout='wide')

st.markdown(
    """
    <style>
    body {
        background-color: #181818;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with open('./data/wave.css') as f:
        css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
########################################################
define_sidebar()
st.title('Transfers')

col1, col2 = st.columns([6,3])

with col1:
    st.write('Table ordered by most transferred in this GW.')
    
    def display_frame(df):
        '''display dataframe with all float columns rounded to 1 decimal place'''
        float_cols = df.select_dtypes(include='float64').columns.values
        st.dataframe(df.style.format(subset=float_cols, formatter='{:.2f}'))
    
    # Most transferred in this GW df
    ele_types_data = get_bootstrap_data()['element_types']
    ele_types_df = pd.DataFrame(ele_types_data)
    
    teams_data = get_bootstrap_data()['teams']
    teams_df = pd.DataFrame(teams_data)
    
    ele_data = get_bootstrap_data()['elements']
    ele_df = pd.DataFrame(ele_data)
    
    ele_df['element_type'] = ele_df['element_type'].map(ele_types_df.set_index('id')['singular_name_short'])
    ele_df['team'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])
    
    ele_df['full_name'] = ele_df['first_name'] + ' ' + ele_df['second_name'] + ' (' + ele_df['team'] + ')'
    
    rn_cols = {'web_name': 'Name', 'team': 'Team', 'element_type': 'Pos', 
               'event_points': 'GW_Pts', 'total_points': 'T_Pts',
               'now_cost': 'Price', 'selected_by_percent': 'TSB%',
               'minutes': 'Mins', 'goals_scored': 'GS', 'assists': 'A',
               'penalties_missed': 'Pen_Miss', 'clean_sheets': 'CS',
               'goals_conceded': 'GC', 'own_goals': 'OG',
               'penalties_saved': 'Pen_Save', 'saves': 'S',
               'yellow_cards': 'YC', 'red_cards': 'RC', 'bonus': 'B', 'bps': 'BPS',
               'value_form': 'Value', 'points_per_game': 'PPG', 'influence': 'I',
               'creativity': 'C', 'threat': 'T', 'ict_index': 'ICT',
               'influence_rank': 'I_Rank', 'creativity_rank': 'C_Rank',
               'threat_rank': 'T_Rank', 'ict_index_rank': 'ICT_Rank',
               'transfers_in_event': 'T_In', 'transfers_out_event': 'T_Out',
               'transfers_in': 'T_In_Total', 'transfers_out': 'T_Out_Total'}
    ele_df.rename(columns=rn_cols, inplace=True)
    ele_df['Price'] = ele_df['Price']/10
    
    ele_cols = ['Name', 'Team', 'Pos', 'GW_Pts', 'T_Pts', 'Price', 'TSB%',
                'T_In', 'T_Out', 'T_In_Total', 'T_Out_Total', 'full_name']
    
    ele_df = ele_df[ele_cols]
    
    ele_df['T_+/-'] = ele_df['T_In'] - ele_df['T_Out']
    
    total_mans = get_total_fpl_players()
    
    ele_df['TSB%'] = ele_df['TSB%'].astype(float)/100
    ele_df['%_+/-'] = ele_df['T_+/-']/total_mans
    
    ele_df.set_index('Name', inplace=True)
    ele_df.sort_values('T_+/-', ascending=False, inplace=True)
    
    ordered_names = ele_df['full_name'].tolist()
    
    trans_df = ele_df.copy()
    
    trans_df.drop('full_name', axis=1, inplace=True)
    
    st.dataframe(trans_df.style.format({'Price': '£{:.1f}',
                                      'TSB%': '{:.1%}',
                                      '%_+/-': '{:.2%}'}))

with col2:
    st.write('Table ordered by biggest price increase this PL Season.')
    prices_df = pd.read_csv('./data/player_prices.csv')
    prices_df.set_index('Player', inplace=True)
    st.dataframe(prices_df.style.format({'Start': '£{:.1f}',
                                         'Now': '£{:.1f}',
                                         '+/-': '£{:.1f}'}))
    
    
# Graph of ownership over time for a specific player, two y-axis (transfers in and price?)
#ordered_names = [name for num, name in get_player_id_dict(web_name=False).items()]

#ele_df['full_name'] = ele_df.reset_index()['Name']

#ordered_names = [.tolist() + ' (' + ele_df['Team'] + ')']

col1, col2 = st.columns([2,4])
with col1:
    selected_player = st.selectbox(label="Select Player", options=ordered_names)

def collate_hist_df_from_name(ele_df, player_name):
    player_df = ele_df.loc[ele_df['full_name'] == player_name]
    full_player_dict = get_player_id_dict(order_by_col='total_points', web_name=False)
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    p_data = get_player_data(str(p_id[0]))
    p_df = pd.DataFrame(p_data['history'])
    p_df.loc[p_df['was_home'] == True, 'result'] = p_df['team_h_score']\
        .astype(str) + '-' + p_df['team_a_score'].astype(str)
    p_df.loc[p_df['was_home'] == False, 'result'] = p_df['team_a_score']\
            .astype(str) + '-' + p_df['team_h_score'].astype(str)
    col_rn_dict = {'round': 'GW', 'opponent_team': 'vs',
                   'total_points': 'Pts', 'minutes': 'Mins',
                   'goals_scored': 'GS', 'assists': 'A', 'clean_sheets': 'CS',
                   'goals_conceded': 'GC', 'own_goals': 'OG',
                   'penalties_saved': 'Pen_Save',
                   'penalties_missed': 'Pen_Miss', 'yellow_cards': 'YC',
                   'red_cards': 'RC', 'saves': 'S', 'bonus': 'B',
                   'bps': 'BPS', 'influence': 'I', 'creativity': 'C',
                   'threat': 'T', 'ict_index': 'ICT', 'value': 'Price',
                   'selected': 'SB', 'transfers_in': 'Tran_In',
                   'transfers_out': 'Tran_Out'}
    p_df.rename(columns=col_rn_dict, inplace=True)
    col_order = ['GW', 'vs', 'result', 'Pts', 'Mins', 'GS', 'A', 'Pen_Miss',
                 'CS', 'GC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS',
                 'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out']
    p_df = p_df.iloc[:, [p_df.columns.get_loc(col) for col in col_order]]
    p_df['Price'] = p_df['Price']/10
    new_df = pd.DataFrame(data = {'GW': [(p_df['GW'].max() + 1)],
                           'Price': [player_df['Price'].iloc[0]],
                           'Tran_In': [player_df['T_In'].iloc[0]],
                           'Tran_Out': [player_df['T_Out'].iloc[0]],
                           'SB': [p_df['SB'].iloc[-1] + player_df['T_In'].iloc[0] - player_df['T_Out'].iloc[0]]})
    p_df = pd.concat([p_df, new_df])
    # map opponent teams
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    p_df.set_index('GW', inplace=True)
    return p_df

try:
    player_hist_df = collate_hist_df_from_name(ele_df, selected_player)
    min_price = player_hist_df['Price'].min()
    max_price = player_hist_df['Price'].max()
    
    min_sb, max_sb = player_hist_df['SB'].min(), player_hist_df['SB'].max()
    min_gw, max_gw = player_hist_df.index.min(), player_hist_df.index.max()
    
    base = alt.Chart(player_hist_df.reset_index()).encode(
        alt.X('GW', axis=alt.Axis(tickMinStep=1, title='GW'), scale=alt.Scale(domain=[min_gw, max_gw]))
    )
    
    price = base.mark_line(color='red').encode(
        alt.Y('Price',
              axis=alt.Axis(tickMinStep=0.1, title='Price (£)', titleColor='Red'),
              scale=alt.Scale(domain=[min_price-0.2, max_price+0.2]))
    )
    
    sel_by = base.mark_line(color='blue').encode(
        alt.Y('SB',
              axis=alt.Axis(tickMinStep=0.1, title='Selected By', titleColor='Blue'),
              scale=alt.Scale(domain=[0, max_sb+1000000]))
    )
    
    c = alt.layer(price, sel_by).resolve_scale(y='independent')
    st.altair_chart(c, use_container_width=True)
    
    
    player_hist_df['T_+/-'] = player_hist_df['Tran_In'] - player_hist_df['Tran_Out']
    min_tran, max_tran = player_hist_df['T_+/-'].min(), player_hist_df['T_+/-'].max()
    
    tran_range = max_tran - min_tran
    
    c = alt.Chart(player_hist_df.reset_index()).mark_line().encode(
        x=alt.X('GW', axis=alt.Axis(tickMinStep=1, title='GW'), scale=alt.Scale(domain=[min_gw, max_gw])),
        y=alt.Y('T_+/-', axis=alt.Axis(tickMinStep=0.1, title='Transfers Total'), scale=alt.Scale(domain=[min_tran-(tran_range*0.1), max_tran+(tran_range*0.1)])),
        ).properties(
            height=400)
    st.altair_chart(c, use_container_width=True)

#    last_gw = player_hist_df.index.max()  

    # Get the previous game week
#      previous_gw = last_gw - 1  

#      price_difference = player_hist_df.loc[last_gw, 'Price'] - player_hist_df.loc[previous_gw, 'Price']

#      st.write(price_difference)


except KeyError as e:
    st.write(f"An error occurred: {e}. Please wait for the Season to begin before viewing transfer data on individual players.")
    
    