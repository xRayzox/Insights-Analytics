import streamlit as st
import pandas as pd
import sys
import os
from io import BytesIO
import requests
import altair as alt
import matplotlib
from matplotlib import pyplot as plt
from mplsoccer import Bumpy
from highlight_text import fig_text
from matplotlib.offsetbox import OffsetImage
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import urllib.request

# Set up Streamlit page
st.set_page_config(page_title='PL Table', page_icon=':sports-medal:', layout='wide')

# Define sidebar and title
from fpl_utils import define_sidebar
define_sidebar()
st.title('Premier League Table')

# Load images from URL
@st.cache_data
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    return Image.open(BytesIO(image_data)).convert("RGBA")

# Define color mappings
def form_color(form):
    color_mapping = {'W': '#28a745', 'D': '#ffc107', 'L': '#dc3545'}
    return [color_mapping[char] for char in form if char in color_mapping]

# Plotting functions
def custom_plot_fn_form(ax: plt.Axes, val):
    colors = form_color(val)
    spacing = 0.2
    for i, (char, color) in enumerate(zip(val, colors)):
        x_pos = 0.5 + (i - (len(val) - 1) / 2) * spacing
        ax.text(x_pos, 0.5, char, fontsize=14, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.5))

# Load league data
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()

# Prepare fixture DataFrame
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)].rename(columns=lambda x: f'GW{x}')
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df).reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)

# Load team data
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + "@x2.png"
teams_df['logo_image'] = teams_df['logo_url'].apply(load_image_from_url)
team_logo_mapping = pd.Series(teams_df['logo_image'].values, index=teams_df['short_name']).to_dict()
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

# Get fixture data
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_df.columns
    merged_dict = {k: [] for k in [1.5, 2.5, 3.5, 4.5]}
    for col in new_fdr_df.columns:
        values = new_fdr_df[col].tolist()
        strings = new_fixt_df[col].tolist()
        value_dict = {v: [] for v in values}
        for v, s in zip(values, strings):
            value_dict[v].append(s)
        for key, value in value_dict.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    return {k: list(set(v)) for k, v in merged_dict.items()}

home_away_dict = get_home_away_str_dict()

# Color fixtures
def color_fixtures(val):
    color_map = {1: "#257d5a", 2: "#00ff86", 3: "#ebebe4", 4: "#ff005a", 5: "#861d46"}
    for key in color_map:
        if val in home_away_dict[key]:
            return color_map[key]
    return "#000000"

def custom_plot_fn(ax: plt.Axes, val):
    ax.text(0.5, 0.5, str(val), fontsize=14, ha='center', va='center', bbox=dict(facecolor=color_fixtures(val), alpha=0.5))

# Define column definitions
col_defs = [
    ColumnDefinition(name="Rank", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="logo_team", textprops={'ha': "center", 'va': "center", 'color': "white"}, plot_fn=image, width=1),
    ColumnDefinition(name="Team", textprops={'ha': "center"}, width=1),
    *[ColumnDefinition(name=col, group="Matches Played", textprops={'ha': "center"}, width=0.5) for col in ['GP', 'W', 'D', 'L']],
    *[ColumnDefinition(name=col, group="Goals", textprops={'ha': "center"}, width=0.5) for col in ['GF', 'GA', 'GD', 'CS']],
    ColumnDefinition(name="Pts", group="Points", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="Pts/Game", group="Points", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="Form", group="Points", textprops={'ha': "center"}, plot_fn=custom_plot_fn_form, width=2),
    *[ColumnDefinition(name=f"{stat}/Game", group="ByGame", textprops={'ha': "center"}, width=1) for stat in ['GF', 'GA', 'CS']],
]

# Add fixture columns dynamically
for gw in range(ct_gw, ct_gw + 3):
    col_defs.append(ColumnDefinition(name=f"GW{gw}", group="Fixtures", textprops={'ha': "center"}, width=1, plot_fn=custom_plot_fn))

# Create the table
fig, ax = plt.subplots(figsize=(20, 20))
fig.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")
table = Table(league_df, column_definitions=col_defs, columns=['logo_team', 'Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'CS', 'Pts', 'Pts/Game', 'Form', 'GF/Game', 'GA/Game', 'CS/Game', f'GW{ct_gw}', f'GW{ct_gw + 1}', f'GW{ct_gw + 2}'], index_col="Rank", row_dividers=True, footer_divider=True, textprops={"fontsize": 14}, ax=ax)

# Set row colors based on rank
row_colors = {
    "top4": "#E1FABC",
    "top6": "#FFFC97",
    "relegation": "#E79A9A",
}
for idx in range(len(league_df)):
    if league_df.iloc[idx]['Rank'] <= 4:
        table.rows[idx].set_facecolor(row_colors["top4"])
    elif league_df.iloc[idx]['Rank'] <= 6:
        table.rows[idx].set_facecolor(row_colors["top6"])
    elif league_df.iloc[idx]['Rank'] >= 18:
        table.rows[idx].set_facecolor(row_colors["relegation"])

# Display table
st.pyplot(fig)

# Team ratings
st.title("Team Offensive / Defensive Ratings")
st.caption("Compare overall, offensive, and defensive strengths of teams.")

# Calculate ratings
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
rating_cols = ['strength_overall_home', 'strength_overall_away', 'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away']
teams_df[rating_cols] = teams_df[rating_cols].astype(float)
teams_df['ovr_rating'] = (teams_df['strength_overall_home'] + teams_df['strength_overall_away']) / 2
teams_df['o_rating'] = (teams_df['strength_attack_home'] + teams_df['strength_attack_away']) / 2
teams_df['d_rating'] = (teams_df['strength_defence_home'] + teams_df['strength_defence_away']) / 2

# Data source selection
model_option = st.selectbox("Data Source", ("Overall", "Home", "Away"))
model_type = model_option.lower() if model_option != "Overall" else ""

# Sort DataFrame
rating_df = teams_df.sort_values("ovr_rating" + (("_" + model_type) if model_type else ""), ascending=False)

# Base64 encoding for logos
@st.cache_data
def load_and_convert_image(url):
    try:
        with urllib.request.urlopen(url) as response:
            image = Image.open(response).convert("RGBA")
        output = BytesIO()
        image.save(output, format='PNG')
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading image from URL {url}: {str(e)}")
        return None

rating_df['logo_data'] = rating_df['code'].apply(lambda x: load_and_convert_image(f"https://resources.premierleague.com/premierleague/badges/70/t{x}@x2.png"))

# Create rating table
st.subheader(f"Teams Sorted by {model_option} Ratings")
st.table(rating_df[['name', 'ovr_rating', 'o_rating', 'd_rating', 'logo_data']].style.format({"logo_data": lambda x: f'<img src="{x}" width="30" height="30"/>'}))
