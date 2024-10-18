import streamlit as st
import datetime as dt
import altair as alt
import pandas as pd
import requests
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.patches as patches 
from PIL import Image
from urllib.request import urlopen
import random
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath




pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data, get_manager_history_data, get_manager_team_data,
    get_manager_details, get_player_data, get_current_season
)
from fpl_utils import (
    define_sidebar, chip_converter
)

from fpl_params import MY_FPL_ID, BASE_URL

st.set_page_config(page_title='Manager', page_icon=':necktie:', layout='wide')
define_sidebar()
st.title('Manager')

def get_total_fpl_players():
    base_resp = requests.get(BASE_URL + 'bootstrap-static/')
    return base_resp.json()['total_players']

# Load bootstrap data
ele_types_data = get_bootstrap_data()['element_types']
ele_types_df = pd.DataFrame(ele_types_data)

teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)

ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)

# Map element types and teams
ele_df['element_type'] = ele_df['element_type'].map(ele_types_df.set_index('id')['singular_name_short'])
ele_df['code'] = ele_df['team'].map(teams_df.set_index('id')['code'])
ele_df['code'] = ele_df['code'].apply(lambda x: f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{x}-110.png")
ele_df['team'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])

col1, col2 = st.columns([3, 2])

with col1:
    fpl_id = st.text_input('Please enter your FPL ID:', MY_FPL_ID)
    if fpl_id:
        try:
            fpl_id = int(fpl_id)
            total_players = get_total_fpl_players()
            if fpl_id <= 0:
                st.write('Please enter a valid FPL ID.')
            elif fpl_id <= total_players:
                manager_data = get_manager_details(fpl_id)
                manager_name = f"{manager_data['player_first_name']} {manager_data['player_last_name']}"
                manager_team = manager_data['name']
                season = get_current_season()
                st.write(f'Displaying {season} GW data for {manager_name}\'s Team ({manager_team})')
                
                man_data = get_manager_history_data(fpl_id)
                current_df = pd.DataFrame(man_data['current'])
                
                if current_df.empty:
                    st.write('Please wait for Season to start before viewing Manager Data')
                else:
                    chips_df = pd.DataFrame(man_data['chips'])
                    if chips_df.empty:
                        ave_df = pd.DataFrame(get_bootstrap_data()['events'])[['id', 'average_entry_score']]
                        ave_df.columns = ['event', 'Ave']
                        man_gw_hist = pd.DataFrame(man_data['current'])
                        man_gw_hist.sort_values('event', ascending=False, inplace=True)
                        man_gw_hist = man_gw_hist.merge(ave_df, on='event', how='left')
                        rn_cols = {
                            'points': 'GWP', 'total_points': 'OP',
                            'rank': 'GWR', 'overall_rank': 'OR',
                            'bank': '£', 'value': 'TV',
                            'event_transfers': 'TM', 'event': 'Event',
                            'event_transfers_cost': 'TC',
                            'points_on_bench': 'PoB', 'name': 'Chip'
                        }
                        man_gw_hist.rename(columns=rn_cols, inplace=True)
                        man_gw_hist.set_index('Event', inplace=True)
                        man_gw_hist.drop(columns='rank_sort', inplace=True)
                        man_gw_hist['TV'] /= 10
                        man_gw_hist['£'] /= 10
                        man_gw_hist = man_gw_hist[['GWP', 'Ave', 'OP', 'GWR', 'OR', '£', 'TV', 'TM', 'TC', 'PoB']]
                        man_gw_hist['Chip'] = 'None'
                        st.dataframe(man_gw_hist.style.format({'TV': '£{:.1f}', '£': '£{:.1f}'}),
                                     width=800, height=522)
                    else:
                        chips_df['name'] = chips_df['name'].apply(chip_converter)
                        chips_df = chips_df[['event', 'name']]
                        ave_df = pd.DataFrame(get_bootstrap_data()['events'])[['id', 'average_entry_score']]
                        ave_df.columns = ['event', 'Ave']
                        man_gw_hist = pd.DataFrame(man_data['current'])
                        man_gw_hist.sort_values('event', ascending=False, inplace=True)
                        man_gw_hist = man_gw_hist.merge(chips_df, on='event', how='left')
                        man_gw_hist = man_gw_hist.merge(ave_df, on='event', how='left')
                        man_gw_hist['name'].fillna('None', inplace=True)
                        rn_cols = {
                            'event': 'GW', 'points': 'GWP',
                            'total_points': 'OP', 'rank': 'GWR',
                            'overall_rank': 'OR', 'bank': '£',
                            'value': 'TV', 'event_transfers': 'TM',
                            'event_transfers_cost': 'TC',
                            'points_on_bench': 'PoB', 'name': 'Chip'
                        }
                        man_gw_hist.rename(columns=rn_cols, inplace=True)
                        man_gw_hist.set_index('GW', inplace=True)
                        man_gw_hist.drop(columns='rank_sort', inplace=True)
                        man_gw_hist['TV'] /= 10
                        man_gw_hist['£'] /= 10
                        man_gw_hist = man_gw_hist[['GWP', 'Ave', 'OP', 'GWR', 'OR', '£', 'TV', 'TM', 'TC', 'PoB', 'Chip']]
                        st.dataframe(man_gw_hist.style.format({'TV': '£{:.1f}', '£': '£{:.1f}'}),
                                     width=800, height=522)
            else:
                st.write('FPL ID is too high to be a valid ID. Please try again.')
                st.write(f'The total number of FPL players is: {total_players}')
        except ValueError:
            st.write('Please enter a valid FPL ID.')
###############################################################################################################



###############################################################################################################
with col2:
    # st.write(manager_name + '\'s Past Results')
    # st.write('Best ever finish: ')
    col = st.selectbox(
        manager_name + '\'s Past Results', ['Season', 'Pts', 'OR']
        )
    hist_data = get_manager_history_data(fpl_id)
    hist_df = pd.DataFrame(hist_data['past'])
    if len(hist_df) == 0:
        st.write('No previous season history to display.')
    else:
        hist_df.columns=['Season', 'Pts', 'OR']
        if col == 'OR':
            hist_df.sort_values(col, ascending=True, inplace=True)
        else:
            hist_df.sort_values(col, ascending=False, inplace=True)
        hist_df.set_index('Season', inplace=True)
        st.dataframe(hist_df, height=562)
#############################################################################
events_df = pd.DataFrame(get_bootstrap_data()['events'])
complete_df = events_df.loc[events_df['deadline_time'] < str(dt.datetime.now())]
gw_complete_list = sorted(complete_df['id'].tolist(), reverse=True)
fpl_gw = st.selectbox('Team on Gameweek', gw_complete_list)

if fpl_id and gw_complete_list:
        man_picks_data = get_manager_team_data(fpl_id, fpl_gw)
        manager_team_df = pd.DataFrame(man_picks_data['picks'])
        ele_cut = ele_df[['id', 'web_name', 'team', 'element_type','code']].copy()
        ele_cut.rename(columns={'id': 'element'}, inplace=True)
        
        manager_team_df = manager_team_df.merge(ele_cut, how='left', on='element')
        
        # Pull GW data for each player
        gw_players_list = manager_team_df['element'].tolist()
        pts_list = []
        for player_id in gw_players_list:
            test = pd.DataFrame(get_player_data(player_id)['history'])
            pl_cols = ['element', 'opponent_team', 'total_points', 'round']
            test_cut = test[pl_cols]
            test_new = test_cut.loc[test_cut['round'] == fpl_gw]

            if test_new.empty:
                test_new = pd.DataFrame([[player_id, 'BLANK', 0, fpl_gw]], columns=pl_cols)

            pts_list.append(test_new)

        pts_df = pd.concat(pts_list)
        manager_team_df = manager_team_df.merge(pts_df, how='left', on='element')

        # Calculate total points based on multiplier and captaincy
        manager_team_df.loc[((manager_team_df['multiplier'] == 2) |
                             (manager_team_df['multiplier'] == 3)),
                            'total_points'] *= manager_team_df['multiplier']
        
        manager_team_df.loc[(manager_team_df['is_captain'] == True) & (manager_team_df['multiplier'] == 2),
                            'web_name'] += ' (C)'
        
        manager_team_df.loc[manager_team_df['multiplier'] == 3,
                            'web_name'] += ' (TC)'
        
        manager_team_df.loc[manager_team_df['is_vice_captain'] == True,
                            'web_name'] += ' (VC)'
        
        manager_team_df.loc[manager_team_df['multiplier'] != 0, 'Played'] = True

        # Fill NaN values
        manager_team_df['Played'] = manager_team_df['Played'].fillna(False)

        # Convert to appropriate dtypes after filling NaN values
        manager_team_df = manager_team_df.infer_objects()

        # Select relevant columns and rename them
        manager_team_df = manager_team_df[['web_name', 'element_type', 'team', 'opponent_team', 'total_points', 'Played','code']]
        rn_cols = {
            'web_name': 'Player', 'element_type': 'Pos',
            'team': 'Team', 'opponent_team': 'vs',
            'total_points': 'GWP'
        }
        manager_team_df.rename(columns=rn_cols, inplace=True)
        manager_team_df.set_index('Player', inplace=True)

        # Map teams
        manager_team_df['vs'] = manager_team_df['vs'].map(teams_df.set_index('id')['short_name'])
        manager_team_df['vs'] = manager_team_df['vs'].fillna('BLANK')
        
        test = manager_team_df.reset_index()


        # Define the figure size
        fig_size = (8, 8)  # Set to desired size (width, height)

        # Create a vertical pitch with specified size
        pitch = VerticalPitch(pitch_color='grass', line_color='white', stripe=True, corner_arcs=True, half=True)
        fig, ax = pitch.draw(figsize=fig_size, tight_layout=False)  # Draw the pitch

        # Extract pitch dimensions from the figure
        pitch_length = fig.get_figheight() * 10  # Scale as needed (10 is a scaling factor)
        pitch_width = fig.get_figwidth() * 10  # Scale as needed

        # Define placements for each position zone
        zone_height = pitch_length / 6  

        # Position calculations (optimized)
        positions = {
            'GKP': pitch_length + 2.5 * zone_height,
            'DEF': pitch_length + 1.5 * zone_height,
            'MID': pitch_length + 1/3 * zone_height,
            'FWD': pitch_length - zone_height
        }
        df=test[test['Played']==True]
        # Loop through DataFrame and place images
        for index, row in df.iterrows():
            IMAGE_URL = row['code']
            image = Image.open(urlopen(IMAGE_URL))

            pos = row['Pos']
            if pos == 'GKP':
                y_image = positions['GKP']
                x_image = pitch_width / 2  # Centered horizontally
            elif pos == 'DEF':
                y_image = positions['DEF']
                num_def = len(df[df['Pos'] == 'DEF'])  # Number of DEF players
                x_image = (pitch_width / (num_def + 1)) * (index % num_def + 1)  # Centered and distributed
            elif pos == 'MID':
                y_image = positions['MID']
                num_mid = len(df[df['Pos'] == 'MID'])  # Number of MID players
                x_image = (pitch_width / (num_mid + 1)) * (index % num_mid + 1)  # Centered and distributed
            elif pos == 'FWD':
                y_image = positions['FWD']
                num_fwd = len(df[df['Pos'] == 'FWD'])  # Number of FWD players
                x_image = (pitch_width / (num_fwd + 1)) * (index % num_fwd + 1)  # Centered and distributed

            # Draw the image on the pitch
            ax_image = pitch.inset_image(y_image, x_image, image, height=10, ax=ax)
            # Add a rounded rectangle behind the player's name
            player_name = row['Player']  # Assuming the DataFrame has a 'Player' column
            fontsize = 8 
            tp = TextPath(0, 0, row['Player'], size=fontsize)  
            bb = tp.get_extents()
            rect_width = bb.width * 1.1 # Add a little padding (10% here)
            rect_height = 2.5 
            rounding = 1 

            # Create a rounded rectangle patch
            rounded_rect = FancyBboxPatch(
                (x_image - rect_width / 2, y_image - rect_height - 8),  
                rect_width,
                rect_height,
                boxstyle=f"round,pad=0.2,rounding_size={rounding}",
                facecolor='white',
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            ax.add_patch(rounded_rect)

            # Add the player's name below the image
            ax.text(x_image, y_image - 8, player_name, fontsize=8, ha='center', color='black')

        # Show the pitch with player images
        plt.show()
        st.pyplot(fig)



###############################################################################

if fpl_id == '':
    st.write('')
else:
    curr_df = pd.DataFrame(get_manager_history_data(fpl_id)['current'])
    if len(curr_df) == 0:
        st.write('')
    else:
        try:
            man_data = get_manager_details(fpl_id)
            curr_df['Manager'] = man_data['player_first_name'] + ' ' + man_data['player_last_name']
            ave_df = pd.DataFrame(get_bootstrap_data()['events'])[['id', 'average_entry_score']]
            ave_df.columns=['event', 'points']
            ave_df['Manager'] = 'GW Average'
            ave_cut = ave_df.loc[ave_df['event'] <= max(curr_df['event'])]
            concat_df = pd.concat([curr_df, ave_cut])
            c = alt.Chart(concat_df).mark_line().encode(
                x=alt.X('event', axis=alt.Axis(tickMinStep=1, title='GW'), scale=alt.Scale(domain=[1, len(curr_df)+1])),
                y=alt.Y('points', axis=alt.Axis(title='GW Points')),
                color='Manager').properties(
                    height=400)
            st.altair_chart(c, use_container_width=True)
        except KeyError:
            st.write('')

def collate_manager_history(fpl_id):
    df = pd.DataFrame(get_manager_history_data(fpl_id)['current'])
    data = get_manager_details(fpl_id)
    df['Manager'] = data['player_first_name'] + ' ' + data['player_last_name']
    return df
