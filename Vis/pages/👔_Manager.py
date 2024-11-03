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
from PIL import Image, ImageDraw, ImageOps
from PIL import Image
from urllib.request import urlopen
import io
import requests

pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data, get_manager_history_data, get_manager_team_data,
    get_manager_details, get_player_data, get_current_season
)
from fpl_utils import (
    define_sidebar, chip_converter,get_total_fpl_players
)
from fpl_league import (
    fetch_league_info,
    get_manager_details,
    get_manager_history_data,
    get_bootstrap_data,

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
ele_df['code'] = ele_df.apply(lambda row: f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{row['code']}.png" 
                              if row['element_type'] == 'GKP' else f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{row['code']}.png", axis=1)
ele_df['team'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])

col1, col2 = st.columns([10, 3])

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
                        st.dataframe(man_gw_hist.style.format({'TV': '£{:.1f}', '£': '£{:.1f}'}),use_container_width=True)
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
                        st.dataframe(man_gw_hist.style.format({'TV': '£{:.1f}', '£': '£{:.1f}'}),use_container_width=True)
            else:
                st.write('FPL ID is too high to be a valid ID. Please try again.')
                st.write(f'The total number of FPL players is: {total_players}')
        except ValueError:
            st.write('Please enter a valid FPL ID.')
###############################################################################################################
session = requests.Session()

def load_image(image_url):
    # Fetch the image
    response = session.get(image_url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    image_data = io.BytesIO(response.content)  # Use BytesIO to handle the image data
    image = Image.open(image_data)  # Open the image with Pillow
    return image


###############################################################################################################
with col2:
    # st.write(manager_name + '\'s Past Results')
    # st.write('Best ever finish: ')
    col = st.selectbox(
        manager_name + '\'s Past Results', ['Season', 'Pts', 'OR']
        )
    # Add space before the table
    st.write(' ')  # or st.write('\n') for an empty line
    st.write(' ')

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
        st.dataframe(hist_df)
#############################################################################

col4,col5,col6 = st.columns([1,20,1])
############################################


###########################################
with col5:
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    # Define the current epoch timestamp and add 1 hour
    current_epoch_plus_1_hour = int((dt.datetime.now() - dt.timedelta(hours=1)).timestamp())
    # Filter events_df based on the 'deadline_time' as epoch
    complete_df = events_df.loc[events_df['deadline_time_epoch'] < current_epoch_plus_1_hour]
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
            
            # Reset index of the manager's team DataFrame
            test = manager_team_df.reset_index()

            # Define the figure size
            fig_size = (8, 8)  # Set to desired size (width, height)

            # Create a vertical pitch with specified size
            pitch = VerticalPitch(
                pitch_color='grass', 
                line_color='white', 
                stripe=True, 
                corner_arcs=True, 
                half=True,
                pad_bottom=20
            )

            fig, ax = pitch.draw(figsize=fig_size, tight_layout=False)  # Draw the pitch

            # Extract pitch dimensions from the figure
            pitch_length = fig.get_figheight() * 10  # Scaling factor
            pitch_width = fig.get_figwidth() * 10  # Scaling factor

            # Define placements for each position zone
            zone_height = pitch_length / 6  
            

            # Position calculations
            positions = {
                'GKP': pitch_length + 2.7 * zone_height,
                'DEF': pitch_length + 1.5 * zone_height,
                'MID': pitch_length + 1/3 * zone_height,
                'FWD': pitch_length - zone_height
            }

            # Filter DataFrame for players who played
            df = test[test['Played'] == True]
            total_gwp = df['GWP'].sum()

            rect = plt.Rectangle(
                (0, pitch_length + 1.8 * zone_height),  # Bottom left corner of the rectangle
                pitch_width / 5,                         # Width of the rectangle
                pitch_width / 5,                         # Height of the rectangle
                color=(55/255, 0/255, 60/255),           # Rectangle color (rgb(55, 0, 60))

            )

            # Add rectangle to the plot
            ax.add_patch(rect)

            # Add text to the rectangle
            ax.text(
                0.5 * (pitch_width / 5),                
                (pitch_length + 1.8 * zone_height + (pitch_width / 5) / 2)+3,  
                f'GW{fpl_gw}',          
                fontsize=20,                            
                color='white',                          
                ha='center',                            
                va='center'                             
            )

            # If total_gwp is an RGB color, you can set it like this
            total_gwp_color = (5/255, 250/255, 135/255)  # rgb(5, 250, 135)
            # Assuming total_gwp is just a number or a string, add it separately
            ax.text(
                0.5 * (pitch_width / 5),               
                (pitch_length + 1.8 * zone_height + (pitch_width / 5) / 2) - 2,  
                str(total_gwp),                        
                fontsize=20,                         
                color=total_gwp_color,              
                ha='center',                           
                va='center'                           
            )

            # Function to draw player images and details
            def draw_players(df, positions):
                for index, row in df.iterrows():
                    IMAGE_URL = row['code']
                    image = load_image(IMAGE_URL)



                    pos = row['Pos']
                    num_players = len(df[df['Pos'] == pos])  # Number of players in this position
                    y_image = positions[pos]
                    x_image = (pitch_width / (num_players + 1)) * (index % num_players + 1) if num_players > 1 else pitch_width / 2

                    # Draw the player image on the pitch
                    pitch.inset_image(y_image, x_image, image, height=9, ax=ax)

                    # Draw player's name and GWP points
                    draw_player_details(ax, row, x_image, y_image)

            # Function to draw player details
            def draw_player_details(ax, row, x_image, y_image):
                player_name = row.Player  # Access using attribute-style access
                gwp_points = row.GWP  

                # Calculate text dimensions
                tp = TextPath((0, 0), player_name, size=2)
                rect_width = tp.get_extents().width  # Add padding
                rect_height = 1

                # Draw player's name rectangle
                rounded_rect = FancyBboxPatch(
                    (x_image - rect_width / 2, y_image - rect_height - 5),
                    rect_width,
                    rect_height,
                    facecolor='white',
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.8
                )
                ax.add_patch(rounded_rect)

                # Draw GWP rectangle
                gwp_rect_y = y_image - rect_height - 7  # Adjust y position for GWP rectangle
                gwp_rect = FancyBboxPatch(
                    (x_image - rect_width / 2, gwp_rect_y),
                    rect_width,
                    rect_height,
                    facecolor=(55 / 255, 0 / 255, 60 / 255),
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.9
                )
                ax.add_patch(gwp_rect)

                # Add text
                ax.text(x_image, gwp_rect_y + rect_height / 2, f"{gwp_points}", fontsize=7, ha='center', color='white', va='center') 
                ax.text(x_image, y_image - rect_height - 5 + rect_height / 2, player_name, fontsize=7, ha='center', color='black', va='center')

            # Draw players who played
            draw_players(df, positions)

            ############################### Bench Players ##################
            df_bench = test[test['Played'] == False]  # Bench players

            # Define bench position and dimensions
            bench_width = pitch_width
            bench_height = pitch_length / 5.3
            bench_x = pitch_width - bench_width
            bench_y = pitch_length - 3 * zone_height

            # Create a rectangle for the bench area
            bench_rect = FancyBboxPatch(
                (bench_x, bench_y),
                bench_width,
                bench_height,
                boxstyle="round,pad=0.2",
                facecolor='#72cf9f',
                edgecolor='#72cf9f',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(bench_rect)

            # Set the total number of bench slots
            bench_slots = 4
            slot_width = bench_width / bench_slots

            # Function to draw bench players
            def draw_bench_players(df_bench):
                for i, row in enumerate(df_bench.itertuples()):
                    IMAGE_URL = row.code  # Access using attribute-style access
                    image = load_image(IMAGE_URL)


                    # Calculate x position for bench players
                    x_bench = bench_x + (slot_width * (i + 0.5))
                    y_bench = bench_y + (bench_height / 2) + 2

                    # Place player images in the bench area
                    pitch.inset_image(y_bench, x_bench, image, height=9, ax=ax)  # Smaller image size for bench players

                    # Draw player details on bench
                    draw_player_details(ax, row, x_bench, y_bench)

            # Draw bench players
            draw_bench_players(df_bench)
            # Show the plot
            plt.show()
            st.pyplot(fig)

###############################################################################

################################################################################
if fpl_id == '':
    st.write('No FPL ID provided.')
else:
    try:
        leagues = manager_data['leagues']['classic']
        filtered_leagues = [league for league in leagues if league.get('league_type') == 'x']
        leagues_names_ids = [(league['id'], league['name']) for league in filtered_leagues]

        # Check if there are any leagues to select
        if not leagues_names_ids:
            st.write('No leagues available.')
        else:
            # Extract IDs and Names
            league_ids, league_names = zip(*leagues_names_ids)

            # Streamlit selectbox
            selected_league_id = st.selectbox('List of Leagues', league_ids,
                                                format_func=lambda x: league_names[league_ids.index(x)])
            ss = fetch_league_info(selected_league_id)
            teams_managers = [(entry['team_id'], entry['name'], entry['player_name']) for entry in ss['entries']]

            # Prepare options for the multiselect
            options = [entry[2] for entry in teams_managers]  # Player names
            team_ids = [entry[0] for entry in teams_managers]  # Team IDs
            default = options[:5]
            # Show teams selection
            selected_teams = st.multiselect(
                label='Show teams',
                options=options,
                default=default,
                format_func=lambda x: x,
            )

            # Extract team IDs from the selected teams
            selected_team_ids = [team_ids[options.index(team)] for team in selected_teams]

            # Filter teams_managers to only include selected teams
            filtered_teams_managers = [manager for manager in teams_managers if manager[0] in selected_team_ids]

            # Initialize an empty list to store individual manager data
            manager_data = []

            # Using caching for manager details and history data
            @st.cache_data
            def get_all_managers_data(filtered_teams):
                data = []
                for team_id, name, player_name in filtered_teams:
                    man_data = get_manager_details(team_id)
                    curr_df = pd.DataFrame(get_manager_history_data(team_id)['current'])
                    curr_df['Manager'] = f"{man_data['player_first_name']} {man_data['player_last_name']}"
                    data.append(curr_df)
                return pd.concat(data)

            # Fetch manager data
            final_df = get_all_managers_data(filtered_teams_managers)

            # Create average points DataFrame
            ave_df = pd.DataFrame(get_bootstrap_data()['events'])[['id', 'average_entry_score']]
            ave_df.columns = ['event', 'points']
            ave_df['Manager'] = 'GW Average'

            # Combine current manager's data and average data
            max_event = max(final_df['event']) if not final_df.empty else 0
            ave_cut = ave_df.loc[ave_df['event'] <= max_event]
            final_df = pd.concat([final_df, ave_cut])

            # Create the chart
            c = alt.Chart(final_df).mark_line().encode(
                x=alt.X('event', axis=alt.Axis(tickMinStep=1, title='GW'), 
                         scale=alt.Scale(domain=[1, max_event + 1])),
                y=alt.Y('points', axis=alt.Axis(title='GW Points')),
                color='Manager'
            ).properties(height=400)

            # Display the chart
            st.altair_chart(c, use_container_width=True)
    except KeyError as e:
        st.write(f'Error retrieving data: {e}. Please try again.')
    except Exception as e:
        st.write(f'An unexpected error occurred: {e}. Please try again.')
