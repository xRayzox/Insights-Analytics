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
# Load data using provided functions
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])

# Get the minimum and maximum game weeks
gw_min = min(events_df['id'])
gw_max = max(events_df['id'])

# Get the current game week
ct_gw = get_current_gw()

# Reset index for dataframes
fixt = team_fixt_df.reset_index()
drf = team_fdr_df.reset_index()
ga = team_ga_df.reset_index()
gf = team_gf_df.reset_index()

# Rename the first column to 'Team'
fixt.rename(columns={0: 'Team'}, inplace=True)
drf.rename(columns={0: 'Team'}, inplace=True)
ga.rename(columns={0: 'Team'}, inplace=True)
gf.rename(columns={0: 'Team'}, inplace=True)

# Retrieve team logos
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

# Create FDR matrix directly from 'drf' DataFrame
fdr_matrix = drf.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)  # Convert FDR values to integers

# Define the custom color mappings
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

ga_gf_colors = {
    0.0: ("#147d1b", "white"),
    0.5: ("#00ff78", "black"),
    1.0: ("#caf4bd", "black"),
    1.5: ("#eceae6", "black"),
    2.0: ("#fa8072", "black"),
    2.5: ("#ff0057", "white"),
    3.0: ("#920947", "white"),
}

# Function to get the appropriate data for the selected metric
def get_selected_data(metric, slider1, slider2):
    if metric == "Fixture Difficulty Rating (FDR)":
        filtered_fdr_matrix = fdr_matrix[(fdr_matrix['GameWeek'] >= slider1) & (fdr_matrix['GameWeek'] <= slider2)]
        pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')
        pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns]
        return pivot_fdr_matrix
    elif metric == "Average Goals Against (GA)":
        ga_matrix = ga.melt(id_vars='Team', var_name='GameWeek', value_name='GA')
        ga_matrix['GA'] = ga_matrix['GA'].astype(float)
        filtered_ga_matrix = ga_matrix[(ga_matrix['GameWeek'] >= slider1) & (ga_matrix['GameWeek'] <= slider2)]
        pivot_ga_matrix = filtered_ga_matrix.pivot(index='Team', columns='GameWeek', values='GA')
        pivot_ga_matrix.columns = [f'GW {col}' for col in pivot_ga_matrix.columns]
        return pivot_ga_matrix
    elif metric == "Average Goals For (GF)":
        gf_matrix = gf.melt(id_vars='Team', var_name='GameWeek', value_name='GF')
        gf_matrix['GF'] = gf_matrix['GF'].astype(float)
        filtered_gf_matrix = gf_matrix[(gf_matrix['GameWeek'] >= slider1) & (gf_matrix['GameWeek'] <= slider2)]
        pivot_gf_matrix = filtered_gf_matrix.pivot(index='Team', columns='GameWeek', values='GF')
        pivot_gf_matrix.columns = [f'GW {col}' for col in pivot_gf_matrix.columns]
        return pivot_gf_matrix

# Define coloring functions
def color_fdr(value, team):
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center; content: "{team} ({value})";'
    return ''  # Default styling

def color_ga_gf(value, team):
    rounded_value = round(value, 2)
    closest_key = min(ga_gf_colors, key=lambda x: abs(x - rounded_value))
    background_color, text_color = ga_gf_colors[closest_key]
    return f'background-color: {background_color}; color: {text_color}; text-align: center; content: "{team} ({value})";'

# User interface elements
selected_display = st.selectbox("Choose Display:", ['📊Fixture Difficulty Rating', '⚽Average Goals Against', '⚽Average Goals For'])
slider1, slider2 = st.slider('Gameweek Range:', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)
selected_metric = "Fixture Difficulty Rating (FDR)" if selected_display == '📊Fixture Difficulty Rating' else "Average Goals Against (GA)" if selected_display == '⚽Average Goals Against' else "Average Goals For (GF)"

# Retrieve and display the selected data
selected_data = get_selected_data(selected_metric, slider1, slider2)

# Apply styling based on the selected metric
if selected_metric == "Fixture Difficulty Rating (FDR)":
    styled_table = selected_data.style.applymap(lambda value: color_fdr(value, selected_data.index))  # Add team name
else:
    styled_table = selected_data.style.applymap(lambda value: color_ga_gf(value, selected_data.index))  # Add team name

# Display the title and styled table
st.markdown(f"**{selected_metric} for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**", unsafe_allow_html=True)
st.write(styled_table)

# Display legend based on selected metric
with st.sidebar:
    if selected_metric == "Fixture Difficulty Rating (FDR)":
        st.markdown("**Legend (FDR):**")
        for fdr, (bg_color, font_color) in fdr_colors.items():
            label = "Very Easy" if fdr == 1 else "Easy" if fdr == 2 else "Medium" if fdr == 3 else "Difficult" if fdr == 4 else "Very Difficult"
            st.sidebar.markdown(f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>{fdr} - {label}</span>", unsafe_allow_html=True)
    else:
        st.markdown("**Legend (GA/GF):**")
        for ga_gf, (bg_color, font_color) in ga_gf_colors.items():
            st.sidebar.markdown(f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>{ga_gf:.1f} - {ga_gf + 0.4:.1f}</span>", unsafe_allow_html=True)
