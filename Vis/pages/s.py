import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
import sys
import os

# Set up the path to include the FPL directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)

# --- Streamlit Configuration ---
st.set_page_config(layout="wide")  # Use wide layout for better table visualization

# --- Functions ---
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image = Image.open(response).convert("RGBA")
    # Save the image to a temporary file
    temp_filename = f"temp_{os.path.basename(url)}"
    image.save(temp_filename)
    return temp_filename

def highlight_rows(row):
    """Function to apply background color based on rank."""
    if row['Rank'] <= 4:
        return ['background-color: #E1FABC'] * len(row)  # Top 4
    elif row['Rank'] <= 6:
        return ['background-color: #FFFC97'] * len(row)  # Top 6
    elif row['Rank'] >= 18:  # Assuming relegation zone starts at 18
        return ['background-color: #E79A9A'] * len(row)  # Relegation
    else:
        return [''] * len(row)  # Default no color

# --- Data Loading and Processing ---
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df)
float_cols = league_df.select_dtypes(include='float64').columns.values
league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + "@x2.png"
teams_df['logo_image'] = teams_df['logo_url'].apply(load_image_from_url)
team_logo_mapping = pd.Series(teams_df['logo_image'].values, index=teams_df['short_name']).to_dict()
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)

# Calculate and assign rankings in the league DataFrame
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---
st.title("Premier League Table")

# --- Filter Options ---
team_filter = st.sidebar.multiselect(
    "Select Teams",
    options=league_df['Team'].unique(),
    default=league_df['Team'].unique()
)

# Filter the league DataFrame based on the selected teams
filtered_league_df = league_df[league_df['Team'].isin(team_filter)]

# Prepare DataFrame for displaying
display_df = filtered_league_df.copy()
# Convert logo images to HTML for display in DataFrame
display_df['logo_team'] = display_df['logo_team'].apply(lambda x: f'<img src="{x}" width="40" height="40">' if x else '')

# Style the DataFrame with colors and display team logos
styled_df = display_df.style.apply(highlight_rows, axis=1).format({
    'logo_team': lambda x: f'<img src="{x}" width="40" height="40">' if x else '',
    'Pts/Game': "{:.2f}",  # Format points per game
})

# Display the styled DataFrame in Streamlit
st.markdown(styled_df.render(), unsafe_allow_html=True)
