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
sui = team_fixt_df.reset_index()
val = team_fdr_df.reset_index()

# Rename the first column to 'Team'
sui.rename(columns={0: 'Team'}, inplace=True)
val.rename(columns={0: 'Team'}, inplace=True)

# Create FDR matrix directly from 'val' DataFrame
fdr_matrix = val.copy()
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

# Define a color map for FDR values (example colors)
flatui_rev = ['#257d5a', '#00ff86', '#ebebe4', '#ff005a', '#861d46']

# Define a coloring function based on the FDR values using the custom color mapping
def color_fdr(value):
    hash_color = map_float_to_color(value, flatui_rev, 1, 5)  # Assuming FDR values range from 1 to 5
    text_color = get_text_color_from_hash(hash_color)
    return f'background-color: {hash_color}; color: {text_color}; text-align: center;'

# Apply the styling to the pivoted FDR matrix
styled_filtered_fdr_table = pivot_fdr_matrix.style.applymap(color_fdr)

# Display the title with the current game week
st.markdown(
    f"<h2 style='text-align: center;'>Premier League Fixtures - Gameweek {slider1}</h2>",
    unsafe_allow_html=True,
)

# Streamlit app to display the styled table
st.write(styled_filtered_fdr_table)

# Sidebar for the legend
with st.sidebar:
    st.markdown("**Legend:**")
    for fdr, (bg_color, font_color) in zip(range(1, 6), flatui_rev):
        st.sidebar.markdown(
            f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
            f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
            f"</span>",
            unsafe_allow_html=True,
        )
