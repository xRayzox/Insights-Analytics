import streamlit as st
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from plottable import ColumnDefinition, Table
import sys
import os

# Adjust the path to include the FPL directory (assuming it's one level up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

# Now you can import your modules
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

def form_color(form):
    color_mapping = {
        'W': '#28a745',  # Green for Win
        'D': '#ffc107',  # Orange for Draw
        'L': '#dc3545',  # Red for Loss
    }
    return [color_mapping[char] for char in form if char in color_mapping]

# --- Data Loading and Processing ---
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df)
league_df = league_df.reset_index().rename(columns={'team': 'Team'})
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)

# Team logos
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + "@x2.png"
teams_df['logo_image'] = teams_df['logo_url'].apply(load_image_from_url)
team_logo_mapping = pd.Series(teams_df['logo_image'].values, index=teams_df['short_name']).to_dict()
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---
st.title("Premier League Table")

# --- Table Styling ---
bg_color = "#FFFFFF"
text_color = "#000000"

matplotlib.rcParams["text.color"] = text_color
matplotlib.rcParams["font.family"] = "monospace"

# --- Function to Color Fixtures ---
def color_fixtures(val):
    if isinstance(val, str):
        if 'W' in val:
            return "#28a745"  # Win
        elif 'D' in val:
            return "#ffc107"  # Draw
        elif 'L' in val:
            return "#dc3545"  # Loss
    return "#FFFFFF"  # Default color if no match

# --- Define Fixture Colormap ---
fixture_cmap = LinearSegmentedColormap.from_list("fixture_cmap", ["#FFFFFF", "#28a745", "#ffc107", "#dc3545"])

# --- Column Definitions ---
column_definitions = [
    ColumnDefinition(name, cmap=fixture_cmap, formatter=lambda x: "") for name in new_fixt_df.columns
] + [ColumnDefinition("Index", title="", width=1.5, textprops={"ha": "right"})]

# --- Plottable Table ---
fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figsize for Streamlit
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Create the table with the updated column definitions
table = Table(
    league_df,
    column_definitions=column_definitions,
    index_col="Rank",
    row_dividers=True,
    footer_divider=True,
    textprops={"fontsize": 14},  # Adjust fontsize for Streamlit
    col_colors=[bg_color] * len(column_definitions),
)

table.plot(ax=ax)

# --- Show the Table ---
st.pyplot(fig)
