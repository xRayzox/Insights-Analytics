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
fixt = team_fixt_df.reset_index()
drf = team_fdr_df.reset_index()
ga = team_ga_df.reset_index()
gf = team_gf_df.reset_index()

# Rename the first column to 'Team'
fixt.rename(columns={0: 'Team'}, inplace=True)
drf.rename(columns={0: 'Team'}, inplace=True)
ga.rename(columns={0: 'Team'}, inplace=True)
gf.rename(columns={0: 'Team'}, inplace=True)

# Streamlit app
st.title("FPL Fixture Analysis")

# Create sliders for game week selection
slider1, slider2 = st.slider('Gameweek Range:', int(ct_gw), gw_max, [int(ct_gw), int(ct_gw + 10)], 1)

# Define the custom color mapping for FDR values before using it
fdr_colors = {
    1: ("#257d5a", "black"),
    2: ("#00ff86", "black"),
    3: ("#ebebe4", "black"),
    4: ("#ff005a", "white"),
    5: ("#861d46", "white"),
}

# Create FDR matrix and merge with fixtures
fdr_matrix = drf.melt(id_vars="Team", var_name="GameWeek", value_name="FDR")
fdr_matrix["FDR"] = fdr_matrix["FDR"].astype(int)
fdr_matrix["GameWeek"] = fdr_matrix["GameWeek"].str.replace("GW", "").astype(int)
merged_df = fixt.melt(id_vars="Team", var_name="GameWeek", value_name="Fixture")
merged_df["GameWeek"] = merged_df["GameWeek"].str.replace("GW", "").astype(int)
merged_df = pd.merge(merged_df, fdr_matrix, on=["Team", "GameWeek"], how="left")

# Define a function to apply styling to fixtures based on FDR
def style_fixture(row):
    fdr = row["FDR"]
    fixture = row["Fixture"]
    if fdr in fdr_colors:
        background_color, text_color = fdr_colors[fdr]
        return (
            f'<span style="background-color: {background_color}; color: {text_color}; padding: 2px 5px; border-radius: 3px;">'
            f"{fixture} ({fdr})"  # Display fixture with FDR in parentheses
            f"</span>"
        )
    else:
        return fixture

# Apply styling and create the final DataFrame for display
merged_df["StyledFixture"] = merged_df.apply(style_fixture, axis=1)
final_df = (
    merged_df[
        (merged_df["GameWeek"] >= slider1) & (merged_df["GameWeek"] <= slider2)
    ]
    .pivot(index="Team", columns="GameWeek", values="StyledFixture")
    .fillna("")
)
final_df.columns = [f"GW {col}" for col in final_df.columns]

# Display the styled table
st.markdown(
    f"**Fixture Difficulty Rating (FDR) for the Next {slider2-slider1+1} Gameweeks (Starting GW {slider1})**",
    unsafe_allow_html=True,
)

# Use st.write to display the DataFrame with HTML styling
st.write(final_df.style.set_properties(**{"text-align": "center"}), unsafe_allow_html=True)

# FDR Legend
with st.sidebar:
    st.markdown("**Legend (FDR):**")
    for fdr, (bg_color, font_color) in fdr_colors.items():
        st.sidebar.markdown(
            f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
            f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
            f"</span>",
            unsafe_allow_html=True,
        )