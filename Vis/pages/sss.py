import streamlit as st
import pandas as pd
import os
import sys

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
)

# Load data using provided functions with error handling
try:
    team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
except Exception as e:
    st.error(f"Error loading data: {e}")
    sys.exit(1)

# Get the current game week
ct_gw = get_current_gw()

# Prepare fixture data
fx = team_fixt_df.reset_index()
fx.rename(columns={0: 'Team'}, inplace=True)
fx_matrix = fx.melt(id_vars='Team', var_name='GameWeek', value_name='Team_Away')

# Prepare FDR data
drf = team_fdr_df.reset_index()
drf.rename(columns={0: 'Team'}, inplace=True)
fdr_matrix = drf.melt(id_vars='Team', var_name='GameWeek', value_name='FDR')

# Merge fixture data with FDR data
combined_matrix = pd.merge(fx_matrix, fdr_matrix, on=['Team', 'GameWeek'])

# Define the custom color mappings for FDR
fdr_colors = {
    1: ("#257d5a", "black"),    # Very Easy
    2: ("#00ff86", "black"),    # Easy
    3: ("#ebebe4", "black"),    # Medium
    4: ("#ff005a", "white"),    # Difficult
    5: ("#861d46", "white"),    # Very Difficult
}

# Define FDR coloring function
def color_fdr(value):
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    return ''  # Default styling

# Pivot the combined matrix for display
display_matrix = combined_matrix.pivot(index='Team', columns='GameWeek', values='Team_Away')

# Create a mapping of FDR values for styling
fdr_values = combined_matrix.set_index(['Team', 'GameWeek'])['FDR'].unstack().fillna(0)


# Function to apply styling based on FDR values
def fdr_styler(fdr_value):
    return color_fdr(fdr_value)

# Apply styling based on FDR values
styled_display_table = display_matrix.style.apply(
    lambda x: fdr_values.applymap(fdr_styler).values, axis=None
)
# Display the title and styled table
st.markdown("**Fixture Difficulty Rating (FDR) for Away Matches**", unsafe_allow_html=True)
st.write(styled_display_table)

# Optionally display legend in the sidebar
with st.sidebar:
    st.markdown("**Legend (FDR):**")
    for fdr, (bg_color, font_color) in fdr_colors.items():
        label = "Very Easy" if fdr == 1 else "Easy" if fdr == 2 else "Medium" if fdr == 3 else "Difficult" if fdr == 4 else "Very Difficult"
        st.sidebar.markdown(f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>{fdr} - {label}</span>", unsafe_allow_html=True)
