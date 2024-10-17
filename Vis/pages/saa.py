import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

# Import functions from your FPL API and utility modules
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
)
from fpl_utils import (
    define_sidebar,
    get_annot_size,
    map_float_to_color,
    get_text_color_from_hash,
    get_rotation,
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

# Create FDR matrix directly from 'val' DataFrame
fdr_matrix = drf.copy()
fdr_matrix = fdr_matrix.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Convert FDR values to integers
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)

# Create sliders for game week selection
slider1, slider2 = st.slider('Gameweek: ', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)

# Filter FDR matrix based on selected game weeks
filtered_fdr_matrix = fdr_matrix[(fdr_matrix['GameWeek'] >= slider1) & (fdr_matrix['GameWeek'] <= slider2)]

# Pivot the filtered FDR matrix for styling
pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Rename columns for display purposes
pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns]

# Define the custom color mapping for FDR values
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Define a coloring function based on the FDR values using the custom color mapping
def color_fdr(value):
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    else:
        return ''  # No style for undefined values

# Create a selection choice for metrics
selected_metric = st.selectbox(
    "Select Metric:",
    ("Fixture Difficulty Rating (FDR)", "Average Goals Against (GA)", "Average Goals For (GF)")
)

# Create a function to get the appropriate DataFrame based on the selection
def get_selected_data(metric):
    if metric == "Fixture Difficulty Rating (FDR)":
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
        # Correct the pivot function call for GF
        pivot_gf_matrix = filtered_gf_matrix.pivot(index='Team', columns='GameWeek', values='GF') 
        pivot_gf_matrix.columns = [f'GW {col}' for col in pivot_gf_matrix.columns]
        return pivot_gf_matrix

# Get the selected data
selected_data = get_selected_data(selected_metric)

# Display the styled table based on the selected metric
if selected_metric == "Fixture Difficulty Rating (FDR)":
    # Apply the styling for FDR
    styled_table = selected_data.style.map(color_fdr)  # Use .map instead of .applymap
else:
    # Default styling for GA and GF (optional, customize as needed)
    styled_table = selected_data.style.map(lambda x: 'text-align: center;')  # Use .map 

# Display the title with the selected metric
st.markdown(
    f"**{selected_metric} for the Next {slider2-slider1} Gameweeks (Starting GW {slider1})**",
    unsafe_allow_html=True
)

# Streamlit app to display the styled table
st.write(styled_table)

# Sidebar for the legend
with st.sidebar:
    st.markdown("**Legend:**")
    for fdr, (bg_color, font_color) in fdr_colors.items():
        st.sidebar.markdown(
            f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
            f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
            f"</span>",
            unsafe_allow_html=True,
        )