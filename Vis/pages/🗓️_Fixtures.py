import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime, timezone

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

# Create FDR matrix directly from 'val' DataFrame
fdr_matrix = drf.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Convert FDR values to integers
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)

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

    # Filter FDR matrix based on selected game weeks
    filtered_fdr_matrix = fdr_matrix[(fdr_matrix['GameWeek'] >= slider1) & (fdr_matrix['GameWeek'] <= slider2)]

    # Pivot the filtered FDR matrix for styling
    pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

    # Rename columns for display purposes
    pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns].copy()

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
        else:
            return ''

    # Define a coloring function for GA/GF values
    def color_ga_gf(value):
        # Round the value to two decimal places for display and color mapping
        rounded_value = round(value, 2)
        closest_key = min(ga_gf_colors, key=lambda x: abs(x - rounded_value))
        background_color, text_color = ga_gf_colors[closest_key]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'

    # Create a selection choice for metrics
    selected_metric = st.selectbox(
        "Select Metric:",
        ("Fixture Difficulty Rating (FDR)", "Average Goals Against (GA)", "Average Goals For (GF)")
    )

    # Create a function to get the appropriate DataFrame based on the selection
    def get_selected_data(metric):
        if metric == "Fixture Difficulty Rating (FDR)":
            return pivot_fdr_matrix.copy() 
        elif metric == "Average Goals Against (GA)":
            ga_matrix = ga.melt(id_vars='Team', var_name='GameWeek', value_name='GA')
            # Round GA values to 2 decimal places
            ga_matrix['GA'] = ga_matrix['GA'].astype(float).round(2) 
            filtered_ga_matrix = ga_matrix[(ga_matrix['GameWeek'] >= slider1) & (ga_matrix['GameWeek'] <= slider2)]
            pivot_ga_matrix = filtered_ga_matrix.pivot(index='Team', columns='GameWeek', values='GA')
            pivot_ga_matrix.columns = [f'GW {col}' for col in pivot_ga_matrix.columns].copy()
            return pivot_ga_matrix.copy()  
        elif metric == "Average Goals For (GF)":
            gf_matrix = gf.melt(id_vars='Team', var_name='GameWeek', value_name='GF')
            # Round GF values to 2 decimal places
            gf_matrix['GF'] = gf_matrix['GF'].astype(float).round(2) 
            filtered_gf_matrix = gf_matrix[(gf_matrix['GameWeek'] >= slider1) & (gf_matrix['GameWeek'] <= slider2)]
            pivot_gf_matrix = filtered_gf_matrix.pivot(index='Team', columns='GameWeek', values='GF') 
            pivot_gf_matrix.columns = [f'GW {col}' for col in pivot_gf_matrix.columns].copy() 
            return pivot_gf_matrix.copy()

    # Get the selected data
    selected_data = get_selected_data(selected_metric)
    selected_data.index = selected_data.index.map(lambda team: f"<img src='{team_logo_mapping[team]}' style='width:20px; height:20px; vertical-align:middle; margin-right:5px;'/> {team}")
    st.markdown(selected_data.to_html(escape=False), unsafe_allow_html=True)

    # Display the styled table based on the selected metric
    if selected_metric == "Fixture Difficulty Rating (FDR)":
        styled_table = selected_data.style.applymap(color_fdr)  # Use applymap for cell-wise styling

        # Display the title with the selected metric (FDR)
        st.markdown(
            f"**{selected_metric} for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**",
            unsafe_allow_html=True
        )

        # FDR Legend (only if FDR is selected)
        with st.sidebar:
            st.markdown("**Legend (FDR):**")
            for fdr, (bg_color, font_color) in fdr_colors.items():
                st.sidebar.markdown(
                    f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
                    f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
    else:  # For GA and GF
        styled_table = selected_data.style.applymap(color_ga_gf)  # Use applymap for cell-wise styling

        # Display the title with the selected metric (GA or GF)
        st.markdown(
            f"**{selected_metric} for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**",
            unsafe_allow_html=True
        )

        # GA/GF Legend (only if GA or GF is selected)
        with st.sidebar:
            st.markdown("**Legend (GA/GF):**")
            for ga_gf, (bg_color, font_color) in ga_gf_colors.items():
                st.sidebar.markdown(
                    f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
                    f"{ga_gf} Goals"
                    f"</span>",
                    unsafe_allow_html=True,
                )

    # Display the styled table
    st.dataframe(styled_table, height=700)

elif selected_display == '⚔️Premier League Fixtures':
    st.subheader("Premier League Fixtures")

    # Retrieve fixture data from the FPL API
    fixtures = get_fixture_data()

    # Get user's time zone
    user_timezone = get_user_timezone()

    # Sort fixtures by kickoff time
    fixtures_sorted = sorted(fixtures, key=lambda f: f['kickoff_time'])

    for fixture in fixtures_sorted:
        # Convert kickoff time to user's local time
        kickoff_time = datetime.strptime(fixture['kickoff_time'], '%Y-%m-%dT%H:%M:%SZ')
        kickoff_time_local = kickoff_time.replace(tzinfo=timezone.utc).astimezone(user_timezone)

        home_team = fixture['team_h_short']
        away_team = fixture['team_a_short']

        st.markdown(
            f"<img src='{team_logo_mapping[home_team]}' style='width:30px; height:30px; vertical-align:middle; margin-right:10px;'>"
            f"**{home_team}** vs "
            f"**{away_team}**"
            f"<img src='{team_logo_mapping[away_team]}' style='width:30px; height:30px; vertical-align:middle; margin-left:10px;'>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div class='kickoff'>{kickoff_time_local.strftime('%Y-%m-%d %H:%M:%S %Z')}</div>", unsafe_allow_html=True)

