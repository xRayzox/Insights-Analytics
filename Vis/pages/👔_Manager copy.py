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
from functools import lru_cache
from io import BytesIO


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
st.title('Manager')

@st.cache_data()
def get_total_fpl_players():
    base_resp = requests.get(BASE_URL + 'bootstrap-static/')
    return base_resp.json()['total_players']

@st.cache_data(persist="disk")
def load_and_preprocess_fpl_data():
    """Loads and preprocesses FPL bootstrap data."""
    bootstrap_data = get_bootstrap_data()
    ele_types_df = pd.DataFrame(bootstrap_data['element_types'])
    teams_df = pd.DataFrame(bootstrap_data['teams'])
    ele_df = pd.DataFrame(bootstrap_data['elements'])

    ele_df['element_type'] = ele_df['element_type'].map(ele_types_df.set_index('id')['singular_name_short'])
    ele_df['code'] = ele_df.apply(lambda row: f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{row['code']}.png", axis=1)
    ele_df['team'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])
    return ele_types_df, teams_df, ele_df

# Call the function to get the preprocessed DataFrames
ele_types_df, teams_df, ele_df = load_and_preprocess_fpl_data()
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
@st.cache_resource
def load_image(url):
    """Load an image from a URL and cache it."""
    try:
        # Fetch the image from the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        img = Image.open(BytesIO(response.content))  # Open the image
        return img
    except Exception as e:
        st.error(f"Error loading image from {url}: {e}")
        return None

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

col4,col5,col6 = st.columns([1,500,1])
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

            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
            from matplotlib.textpath import TextPath
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            import matplotlib.image as mpimg

            # Load custom background image (or create a blank canvas)
            # Read the image
            from PIL import Image
            import matplotlib.pyplot as plt

            # Read the image
            background_img = Image.open("./data/Pitch.png")

            # Get original dimensions
            img_width, img_height = background_img.size

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size as needed
            ax.imshow(background_img)

            # Set transparent background for Streamlit
            #fig.patch.set_alpha(0)
            ax.set_facecolor('none')

            # Hide axes for a clean display
            ax.axis('off')

            # Replace pitch dimensions with resized image dimensions
            pitch_length = img_height  # Use the resized height
            pitch_width = img_width  # Use the resized width

            # Show the plot (for testing purposes)
            plt.show()


            # Define placements for each position zone based on the new background
            zone_height = pitch_length / 6  # Adjust height zones for positions
            positions = {
                'GKP': pitch_length - 5.5 * zone_height,
                'DEF': pitch_length - 4.5 * zone_height,
                'MID': pitch_length - 3.5 * zone_height,
                'FWD': pitch_length - 2 * zone_height
            }
            def draw_players(df, positions, ax, img_width):
                for index, row in df.iterrows():
                    IMAGE_URL = row['code']
                    image = load_image(IMAGE_URL)
                    pos = row['Pos']
                    num_players = len(df[df['Pos'] == pos])  # Number of players in this position
                    y_image = positions[pos]
                    
                    # Adjust horizontal spacing for players in the same position
                    margin_factor = 0.1  # Adjust this to reduce spacing
                    x_start = img_width * margin_factor
                    x_end = img_width * (1 - margin_factor)
                    x_range = x_end - x_start 
                    x_image = x_start + (x_range / (num_players + 1)) * (index % num_players + 1)*0.1 if num_players > 1 else img_width / 2

                    # Place player image on the background
                    player_image = OffsetImage(image, zoom=0.1)  # Adjust zoom as needed
                    player_box = AnnotationBbox(player_image, (x_image, y_image), frameon=False)
                    ax.add_artist(player_box)

            # Draw players who played
            df_played = test[test['Played'] == True]
            draw_players(df_played, positions, ax, pitch_width)

            # Display in Streamlit
            import streamlit as st
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
