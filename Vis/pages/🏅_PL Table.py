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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image, image
import urllib.request
from PIL import Image
import base64
from io import BytesIO
import tempfile


# Configure Streamlit
st.set_page_config(page_title='PL Table', page_icon=':sports-medal:', layout='wide')

# Pandas options
pd.set_option('future.no_silent_downcasting', True)

# Add FPL module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'FPL')))

# Import FPL functions
from fpl_api_collection import (get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data)
from fpl_utils import define_sidebar

# --- Sidebar ---
define_sidebar()


# --- Title ---
st.title('Premier League Table')

# --- Image and Color Helper Functions ---

def download_image_to_temp(url):
    """Downloads image from URL and saves it to a temporary file, returning the base64 encoded string."""
    if url:
        try:
            with urllib.request.urlopen(url) as response, tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp_file:
                image = Image.open(response).convert("RGBA")
                image.save(tmp_file, format='PNG')
                tmp_file.seek(0)
                return "data:image/png;base64," + base64.b64encode(tmp_file.read()).decode()
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None
    return None

def form_color(form):
    """Returns a list of colors representing the form string (W, D, L)."""
    color_mapping = {'W': '#28a745', 'D': '#ffc107', 'L': '#dc3545'}
    return [color_mapping[char] for char in form if char in color_mapping]

def color_fixtures(val):
    """Returns a color based on the fixture difficulty (FDR)."""
    color_map = {1: "#257d5a", 2: "#00ff86", 3: "#ebebe4", 4: "#ff005a", 5: "#861d46"}
    for key in color_map:
        if val in home_away_dict[key]:
            return color_map[key]
    return "#000000"


# --- Data Loading and Processing ---

league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = [f'GW{col}' for col in new_fixt_df.columns]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)] 

league_df = league_df.join(new_fixt_df).reset_index()
league_df.rename(columns={'team': 'Team', 'index':'Rank'}, inplace=True)
league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = f"https://resources.premierleague.com/premierleague/badges/70/t{teams_df['code']}@x2.png"

teams_df['logo_base64'] = teams_df['logo_url'].apply(download_image_to_temp)

team_logo_mapping = dict(zip(teams_df['short_name'], teams_df['logo_base64']))
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)



# --- Fixture String Processing ---
def get_home_away_str_dict():
    """Processes fixture strings and groups them by FDR."""

    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length + 9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            value_dict.setdefault(value, []).append(string)
        result_dict[col] = value_dict

    merged_dict = {k: [] for k in [1.5, 2.5, 3.5, 4.5]}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            merged_dict.setdefault(key, []).extend(value)

    for k, v in merged_dict.items():
        merged_dict[k] = list(set(v))
    for i in range(1, 6):
        merged_dict.setdefault(i, [])  # Ensure all FDRs 1-5 have entries
    return merged_dict

home_away_dict = get_home_away_str_dict()


# --- Plottable Table Plotting Functions ---
def custom_plot_fn_form(ax: plt.Axes, val):
    """Plots the form string with colored boxes."""
    colors = form_color(val)
    num_chars = len(val)
    spacing = 0.2
    for i, (char, color) in enumerate(zip(val, colors)):
        x_pos = 0.5 + (i - (num_chars - 1) / 2) * spacing
        ax.text(x_pos, 0.5, char, fontsize=14, ha='center', va='center',
                bbox=dict(facecolor=color, alpha=0.5))

def custom_plot_fn(ax: plt.Axes, val):
    """Plots fixture strings with colored boxes."""
    ax.text(0.5, 0.5, str(val), fontsize=14, ha='center', va='center',
            bbox=dict(facecolor=color_fixtures(val), alpha=0.5))

# --- Table Styling ---

bg_color = "#FFFFFF"
text_color = "#000000"

row_colors = {
    "top4": "#E1FABC",
    "top6": "#FFFC97",
    "relegation": "#E79A9A",
    "even": "#E2E2E1",
    "odd": "#B3B0B0"
}

matplotlib.rcParams["text.color"] = text_color
matplotlib.rcParams["font.family"] = "monospace"


# --- Column Definitions for Plottable ---

col_defs = [
    ColumnDefinition("Rank", textprops={'ha': "center"}, width=1),
    ColumnDefinition("logo_team", plot_fn=image, width=1, textprops={'ha': "center", 'va': "center", 'color': "white"}),
    ColumnDefinition("Team", textprops={'ha': "center"}, width=1),
    ColumnDefinition("GP", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("W", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("D", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("L", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("GF", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("GA", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("GD", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("CS", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition("Pts", group="Points", textprops={'ha': "center"}, width=1),
    ColumnDefinition("Pts/Game", group="Points", textprops={'ha': "center"}, width=1),
    ColumnDefinition("Form", group="Points", plot_fn=custom_plot_fn_form, width=2, textprops={'ha': "center"}),
    ColumnDefinition("GF/Game", group="ByGame", textprops={'ha': "center"}, width=1),
    ColumnDefinition("GA/Game", group="ByGame", textprops={'ha': "center"}, width=1),
    ColumnDefinition("CS/Game", group="ByGame", textprops={'ha': "center"}, width=1),
]


for gw in range(ct_gw, ct_gw + 3):
    col_defs.append(ColumnDefinition(f"GW{gw}", group="Fixtures", plot_fn=custom_plot_fn, width=1, textprops={'ha': "center"}))


# --- Create and Style Plottable Table ---

fig, ax = plt.subplots(figsize=(20, 20))
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

table = Table(
    league_df,
    column_definitions=col_defs,
    columns=['logo_team','Team'] + [col.name for col in col_defs[2:] if col.name in league_df.columns], 
    index_col="Rank",
    row_dividers=True,
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    footer_divider=True,
    textprops={"fontsize": 14},
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": .5, "linestyle": "-"},
    ax=ax
)


for idx in range(len(league_df)):
    if league_df.iloc[idx]['Rank'] <= 4:
        table.rows[idx].set_facecolor(row_colors["top4"])
    elif league_df.iloc[idx]['Rank'] <= 6:
        table.rows[idx].set_facecolor(row_colors["top6"])
    elif league_df.iloc[idx]['Rank'] >= 18:
        table.rows[idx].set_facecolor(row_colors["relegation"])

st.pyplot(fig)


# --- Team Ratings Chart ---

st.title("Team Offensive / Defensive Ratings")
st.caption("Compare overall, offensive, and defensive strengths of teams.")

# Calculate ratings
teams_df["ovr_rating_home"] = teams_df["strength_overall_home"]
teams_df["ovr_rating_away"] = teams_df["strength_overall_away"]

teams_df["o_rating_home"] = teams_df["strength_attack_home"]
teams_df["o_rating_away"] = teams_df["strength_attack_away"]

teams_df["d_rating_home"] = teams_df["strength_defence_home"]
teams_df["d_rating_away"] = teams_df["strength_defence_away"]


teams_df["ovr_rating"] = (teams_df["ovr_rating_home"] + teams_df["ovr_rating_away"]) / 2
teams_df["o_rating"] = (teams_df["o_rating_home"] + teams_df["o_rating_away"]) / 2
teams_df["d_rating"] = (teams_df["d_rating_home"] + teams_df["d_rating_away"]) / 2


# User selection
model_option = st.selectbox("Data Source", ("Overall", "Home", "Away"))
model_type = "home" if model_option == "Home" else "away" if model_option == "Away" else ""

# Sort and prepare DataFrame
rating_df = teams_df.sort_values("ovr_rating" + ("_" + model_type if model_type else ""), ascending=False)


for col in ["ovr_rating", "o_rating", "d_rating"]:
    rating_df[col + ("_" + model_type if model_type else "")] = rating_df[col + ("_" + model_type if model_type else "")].astype(float)

# Calculate rating ranges with margins for chart scaling
d_rating_min = teams_df["d_rating" + ("_" + model_type if model_type else "")].min()
d_rating_max = teams_df["d_rating" + ("_" + model_type if model_type else "")].max()
o_rating_min = teams_df["o_rating" + ("_" + model_type if model_type else "")].min()
o_rating_max = teams_df["o_rating" + ("_" + model_type if model_type else "")].max()


x_margin = (d_rating_max - d_rating_min) * 0.05
y_margin = (o_rating_max - o_rating_min) * 0.1
x_domain = [d_rating_min - x_margin, d_rating_max + x_margin]
y_range  = [o_rating_min - y_margin, o_rating_max + y_margin]




# --- Create and Display Altair Chart ---

scatter_plot = (
    alt.Chart(teams_df, height=400, width=800)
    .mark_image(width=30, height=30)
    .encode(
        x=alt.X("d_rating" + ("_" + model_type if model_type else ""), type="quantitative", title="Defensive Rating", scale=alt.Scale(domain=x_domain)),
        y=alt.Y("o_rating" + ("_" + model_type if model_type else ""), type="quantitative", title="Offensive Rating", scale=alt.Scale(domain=y_range)),
        tooltip=[
            alt.Tooltip("name", title="Team"),
            alt.Tooltip("ovr_rating" + ("_" + model_type if model_type else ""), title="Overall Rating", format=".2f"),
            alt.Tooltip("o_rating" + ("_" + model_type if model_type else ""), title="Offensive Rating", format=".2f"),
            alt.Tooltip("d_rating" + ("_" + model_type if model_type else ""), title="Defensive Rating", format=".2f"),
        ],
        url='logo_base64',
    )
)

off_mean_line = alt.Chart(pd.DataFrame({"Mean Offensive Rating": [teams_df["o_rating" + ("_" + model_type if model_type else "")].mean()]})).mark_rule(color="#60b4ff", opacity=0.66).encode(y="Mean Offensive Rating")
def_mean_line = alt.Chart(pd.DataFrame({"Mean Defensive Rating": [teams_df["d_rating" + ("_" + model_type if model_type else "")].mean()]})).mark_rule(color="#60b4ff", opacity=0.66).encode(x="Mean Defensive Rating")


st.altair_chart(scatter_plot + off_mean_line + def_mean_line, use_container_width=True)