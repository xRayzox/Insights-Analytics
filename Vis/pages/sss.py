import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
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
drf = team_fdr_df.reset_index()
drf.rename(columns={0: 'Team'}, inplace=True)

# Retrieve team logos
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

# Create FDR matrix directly from 'drf' DataFrame
fdr_matrix = drf.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)  # Convert FDR values to integers
#####################################
fx=team_fixt_df.reset_index()
fx.rename(columns={0: 'Team'}, inplace=True)
fx_matrix = fx.melt(id_vars='Team', var_name='GameWeek', value_name='Team_Away')



st.write(fx_matrix)
#####################################
 
# Define the custom color mappings for FDR
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Function to get the FDR data within the selected gameweek range
def get_fdr_data(slider1, slider2):
    filtered_fdr_matrix = fdr_matrix[(fdr_matrix['GameWeek'] >= slider1) & (fdr_matrix['GameWeek'] <= slider2)]
    st.write(filtered_fdr_matrix)

    pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')
    pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns]
    return pivot_fdr_matrix

# Define FDR coloring function
def color_fdr(value):
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    return ''  # Default styling

# User interface elements for gameweek range selection
slider1, slider2 = st.slider('Gameweek Range:', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)

# Retrieve and display the FDR data
fdr_data = get_fdr_data(slider1, slider2)
styled_fdr_table = fdr_data.style.applymap(color_fdr)

# Display the title and styled table
st.markdown(f"**Fixture Difficulty Rating (FDR) for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**", unsafe_allow_html=True)
st.write(styled_fdr_table)

# Display FDR legend in the sidebar
with st.sidebar:
    st.markdown("**Legend (FDR):**")
    for fdr, (bg_color, font_color) in fdr_colors.items():
        label = "Very Easy" if fdr == 1 else "Easy" if fdr == 2 else "Medium" if fdr == 3 else "Difficult" if fdr == 4 else "Very Difficult"
        st.sidebar.markdown(f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>{fdr} - {label}</span>", unsafe_allow_html=True)
