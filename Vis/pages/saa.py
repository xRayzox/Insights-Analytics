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
fdr_matrix['FDR'] = fdr_matrix['FDR'].astype(int)  # Convert FDR to integers

# Create sliders for game week selection
slider1, slider2 = st.slider('Gameweek: ', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)

# Filter both DataFrames based on selected game weeks
filtered_fdr_matrix = fdr_matrix[(fdr_matrix['GameWeek'] >= slider1) & (fdr_matrix['GameWeek'] <= slider2)]
filtered_fixt_df = sui[(sui.columns >= slider1) & (sui.columns <= slider2)]

# Pivot the filtered FDR matrix for styling
pivot_fdr_matrix = filtered_fdr_matrix.pivot(index='Team', columns='GameWeek', values='FDR')

# Pivot the fixture DataFrame for display
pivot_fixt_df = filtered_fixt_df.melt(id_vars='Team', var_name='GameWeek', value_name='Opponent')
pivot_fixt_df = pivot_fixt_df.pivot(index='Team', columns='GameWeek', values='Opponent')

# Rename columns for display purposes
pivot_fdr_matrix.columns = [f'GW {col}' for col in pivot_fdr_matrix.columns]
pivot_fixt_df.columns = [f'GW {col}' for col in pivot_fixt_df.columns]

# Define the custom color mapping for FDR values
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Define a coloring function based on the FDR values using the custom color mapping
def color_fdr(row):
    return [f'background-color: {fdr_colors[value][0]}; color: {fdr_colors[value][1]}; text-align: center;'
            for value in pivot_fdr_matrix.loc[row.name]]

# Apply the styling to the fixture DataFrame using FDR colors
styled_filtered_fixt_table = pivot_fixt_df.style.apply(color_fdr, axis=1)

# Display the title with the current game week
st.markdown(
    f"**Fixture Difficulty Rating (FDR) for the Next {slider2 - slider1} Gameweeks (Starting GW{slider1})**",
    unsafe_allow_html=True)

# Streamlit app to display the styled table (using fixture data with FDR colors)
st.write(styled_filtered_fixt_table)

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