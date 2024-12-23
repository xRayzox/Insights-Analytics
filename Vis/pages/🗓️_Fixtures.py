import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime, timezone
import streamlit.components.v1 as components
import pytz
pd.set_option('future.no_silent_downcasting', True)
# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_fixture_data
)
from fpl_utils import (
    
    define_sidebar,
    get_annot_size,
    map_float_to_color,
    get_text_color_from_hash,
    get_rotation,
    get_user_timezone
)

st.set_page_config(page_title='Fixtures', page_icon=':calendar:', layout='wide')
st.markdown(
    """
    <style>
    .kickoff {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Forcing dark mode
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

    # Define the custom color mapping for FDR values
fdr_colors = {
        1: ("#257d5a", "black"),
        2: ("#00ff86", "black"),
        3: ("#ebebe4", "black"),
        4: ("#ff005a", "white"),
        5: ("#861d46", "white"),
    }

    # Define the custom color mapping for GA and GF
ga_gf_colors = {
        0.0: ("#147d1b", "white"),
        0.5: ("#00ff78", "black"),
        1.0: ("#caf4bd", "black"),
        1.5: ("#eceae6", "black"),
        2.0: ("#fa8072", "black"),
        2.5: ("#ff0057", "white"),
        3.0: ("#920947", "white"), 
    }

    # Define a coloring function based on the FDR values
def color_fdr(value):
        if value in fdr_colors:
            background_color, text_color = fdr_colors[value]
            return f'background-color: {background_color}; color: {text_color}; text-align: center;'
        return ''  # Return an empty string for default styling

def fdr_styler(fdr_value):
    return color_fdr(fdr_value)


    # Define a coloring function for GA/GF values
def color_ga_gf(value):
        # Round the value to two decimal places for display and color mapping
        rounded_value = round(value, 2)
        closest_key = min(ga_gf_colors, key=lambda x: abs(x - rounded_value))
        background_color, text_color = ga_gf_colors[closest_key]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'


def fdr_styler_ga(ga_value):
    return color_ga_gf(ga_value)

##########################################################


import streamlit as st
import pandas as pd

@st.cache_data
def load_and_prepare_data():
    # Load data
    team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
    events_df = pd.DataFrame(get_bootstrap_data()['events'])

    gw_min = min(events_df['id'])
    gw_max = max(events_df['id'])

    ct_gw = get_current_gw()
    fixt = team_fixt_df.reset_index()
    drf = team_fdr_df.reset_index()
    ga = team_ga_df.reset_index()
    gf = team_gf_df.reset_index()

    # Rename the first column to 'Team'
    fixt.rename(columns={0: 'Team'}, inplace=True)
    drf.rename(columns={0: 'Team'}, inplace=True)
    ga.rename(columns={0: 'Team'}, inplace=True)
    gf.rename(columns={0: 'Team'}, inplace=True)

    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
    team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

    # Prepare fixture data
    fx = team_fixt_df.reset_index()
    fx.rename(columns={0: 'Team'}, inplace=True)
    fx_matrix = fx.melt(id_vars='Team', var_name='GameWeek', value_name='Team_Away')

    # Create FDR matrix directly from 'val' DataFrame
    fdr_matrix = drf.copy()
    fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')
    # Convert FDR values to integers
    fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)

    combined_matrix_fdr = pd.merge(fx_matrix, fdr_matrix, on=['Team', 'GameWeek'])

    ga_matrix = ga.copy()
    ga_matrix = ga_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='GA')
    ga_matrix['GA'] = ga_matrix['GA'].astype(float)

    combined_matrix_GA = pd.merge(fx_matrix, ga_matrix, on=['Team', 'GameWeek'])

    gf_matrix = gf.copy()
    gf_matrix = gf_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='GF')
    gf_matrix['GF'] = gf_matrix['GF'].astype(float)

    combined_matrix_GF = pd.merge(fx_matrix, gf_matrix, on=['Team', 'GameWeek'])

    return combined_matrix_fdr, combined_matrix_GA, combined_matrix_GF, team_logo_mapping,ct_gw,gw_max,gw_min,events_df

# Use the cached function
combined_matrix_fdr, combined_matrix_GA, combined_matrix_GF, team_logo_mapping,ct_gw,gw_max,gw_min,events_df = load_and_prepare_data()



# Streamlit app
st.title("FPL Fixture Analysis")

# Create a selection choice for the display
with st.sidebar:
    selected_display = st.radio(
        "Select Display:", ['⚔️Premier League Fixtures', '📊Fixture Difficulty Rating']
    )

if selected_display == '📊Fixture Difficulty Rating':
    # Create sliders for game week selection
    slider1, slider2 = st.slider('Gameweek Range:', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)
    # Create a selection choice for metrics
    selected_metric = st.selectbox(
        "Select Metric:",
        ("Fixture Difficulty Rating (FDR)", "Average Goals Against (GA)", "Average Goals For (GF)")
    )
    # Create a function to get the appropriate DataFrame based on the selection
    def get_selected_data(metric):
        if metric == "Fixture Difficulty Rating (FDR)":
            filtered_fdr_matrix = combined_matrix_fdr[(combined_matrix_fdr['GameWeek'] >= slider1) & (combined_matrix_fdr['GameWeek'] <= slider2)]
            fdr_values = filtered_fdr_matrix.set_index(['Team', 'GameWeek'])['FDR'].unstack().fillna(0)
            pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='Team_Away')
            pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns].copy()
            return pivot_fdr_matrix.copy() ,fdr_values
        elif metric == "Average Goals Against (GA)":
            filtered_ga_matrix = combined_matrix_GA[(combined_matrix_GA['GameWeek'] >= slider1) & (combined_matrix_GA['GameWeek'] <= slider2)]
            ga_values = filtered_ga_matrix.set_index(['Team', 'GameWeek'])['GA'].unstack().fillna(0)
            pivot_ga_matrix = filtered_ga_matrix.pivot(index='Team', columns='GameWeek', values='Team_Away')
            pivot_ga_matrix.columns = [f'GW {col}' for col in pivot_ga_matrix.columns].copy() 
            return pivot_ga_matrix.copy(),ga_values
        elif metric == "Average Goals For (GF)":
            filtered_gf_matrix = combined_matrix_GF[(combined_matrix_GF['GameWeek'] >= slider1) & (combined_matrix_GF['GameWeek'] <= slider2)]
            gf_values = filtered_gf_matrix.set_index(['Team', 'GameWeek'])['GF'].unstack().fillna(0)
            pivot_gf_matrix = filtered_gf_matrix.pivot(index='Team', columns='GameWeek', values='Team_Away')
            pivot_gf_matrix.columns = [f'GW {col}' for col in pivot_gf_matrix.columns].copy() 
            return pivot_gf_matrix.copy(),gf_values
        

    # Get the selected data
    selected_data = get_selected_data(selected_metric)
    # Function to handle styling and legend display
    def display_metric_table(selected_metric, selected_data, slider1, slider2, colors):
        styled_table = selected_data[0].style.apply(
            lambda x: selected_data[1].applymap(colors['styler']).values, axis=None
        )
        
        # Display the title with the selected metric
        st.markdown(
            f"**{selected_metric} for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**",
            unsafe_allow_html=True
        )
        
        # Legend
        with st.sidebar:
            st.markdown(f"**Legend ({selected_metric}):**")
            for value, (bg_color, font_color) in colors['legend'].items():
                st.sidebar.markdown(
                    f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
                    f"{value:.1f} - {value + 0.4:.1f}"  # Display the range
                    f"</span>",
                    unsafe_allow_html=True,
                )
        
        return styled_table

    # Mapping for metrics
    metrics_map = {
        "Fixture Difficulty Rating (FDR)": {
            'styler': fdr_styler,
            'legend': fdr_colors
        },
        "Average Goals Against (GA)": {
            'styler': fdr_styler_ga,
            'legend': ga_gf_colors
        },
        "Average Goals For (GF)": {
            'styler': fdr_styler_ga,
            'legend': ga_gf_colors
        }
    }

    # Display the styled table based on the selected metric
    if selected_metric in metrics_map:
        styled_table = display_metric_table(selected_metric, selected_data, slider1, slider2, metrics_map[selected_metric])
        st.write(styled_table)
    else:
        st.error("Selected metric is not recognized.")


###################################
elif selected_display == '⚔️Premier League Fixtures':

    st.write("""
    <script>
        // Function to set a cookie
        function setCookie(name, value, days) {
            var expires = "";
            if (days) {
                var date = new Date();
                date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
                expires = "; expires=" + date.toUTCString();
            }
            document.cookie = name + "=" + (value || "")  + expires + "; path=/";
        }

        var timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        setCookie("timezone", timezone, 365); // Store timezone for a year
    </script>
    """, unsafe_allow_html=True)
    @st.cache_data(ttl=3600) # Cache for an hour to avoid excessive cookie reads
    def get_timezone_from_cookie():
        return st.experimental_get_query_params().get('timezone', ['UTC'])[0]
    # Initialize timezone in session_state if not already present
    if 'timezone' not in st.session_state:
        st.session_state['timezone'] = get_timezone_from_cookie()
    st.session_state['timezone'] = get_timezone_from_cookie()
    timezone = st.session_state['timezone']
    saaaa=get_fixture_data()
    fixtures_df = pd.DataFrame(saaaa)
    fixtures_df.drop(columns='stats', inplace=True)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
    team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()
    fixtures_df = fixtures_df.merge(teams_df[['id', 'logo_url']], left_on='team_h', right_on='id', how='left').rename(columns={'logo_url': 'team_h_logo'})
    fixtures_df = fixtures_df.merge(teams_df[['id', 'logo_url']], left_on='team_a', right_on='id', how='left').rename(columns={'logo_url': 'team_a_logo'})
    fixtures_df['team_a'] = fixtures_df['team_a'].replace(team_name_mapping)
    fixtures_df['team_h'] = fixtures_df['team_h'].replace(team_name_mapping)
    fixtures_df = fixtures_df.drop(columns=['pulse_id'])
    fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
    fixtures_df['local_time'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%A %d %B %Y %H:%M')
    fixtures_df['local_date'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%d %A %B %Y')
    fixtures_df['local_hour'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%H:%M')
    gw_minn = min(fixtures_df['event'])
    gw_maxx = max(fixtures_df['event'])
    if events_df.loc[events_df['is_current'] == True].any().any():
       ct_gw -= 1
    
    gw_minn = int(gw_minn)
    gw_maxx = int(gw_maxx)
    ct_gw = int(ct_gw)
    selected_gw = st.slider('Select Gameweek:', gw_minn, gw_maxx, ct_gw) 
        # --- Display Fixtures for Selected Gameweek ---
    st.markdown(
        f"<h2 style='text-align: center;'>Premier League Fixtures - Gameweek {selected_gw}</h2>",
        unsafe_allow_html=True,
    )

    current_gameweek_fixtures = fixtures_df[fixtures_df['event'] == selected_gw]
    grouped_fixtures = current_gameweek_fixtures.groupby('local_date')

        # Use centered container for fixtures
    with st.container():
        for date, matches in grouped_fixtures:
            st.markdown(f"<h3 style='text-align: center;'>{date}</h3>", unsafe_allow_html=True)
            for _, match in matches.iterrows():
                # Create a fixture box for each match
                with st.container():
                    # Create columns with NO spacing
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        st.markdown(
                            f"<div style='text-align: right;'>"
                            f"{match['team_h']} "
                            f"<img src='{match['team_h_logo']}' style='width:20px; height:20px; vertical-align:middle; margin-left:5px;'/></div>",
                            unsafe_allow_html=True
                        )

                    # --- Column 2: Score/VS (centered) ---
                    with col2:
                        if match['finished']:
                            st.markdown(
                                f"<div style='text-align: center;'>{int(match['team_h_score'])} - {int(match['team_a_score'])}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                "<div style='text-align: center;'>vs</div>",
                                unsafe_allow_html=True
                            )

                    # --- Column 3: Away Team (left-aligned with logo) ---
                    with col3:
                        st.markdown(
                            f"<div style='text-align: left;'>"
                            f"<img src='{match['team_a_logo']}' style='width:20px; height:20px; vertical-align:middle; margin-right:5px;'/>"
                            f"{match['team_a']}</div>",
                            unsafe_allow_html=True
                        )

                    # --- Kickoff Time (centered below) ---
                    if not match['finished']:
                        st.markdown(
                            f"<p style='text-align: center; margin-top: 10px;'>Kickoff: {match['local_hour']}</p>",
                            unsafe_allow_html=True
                        )
define_sidebar()
