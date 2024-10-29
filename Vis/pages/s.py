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
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image
from plottable.plots import image
import sys
import os
pd.set_option('future.no_silent_downcasting', True)

# Adjust the path to include the FPL directory (assuming it's one level up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

# Now you can import your modules
from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)
from fpl_utils import (
    define_sidebar
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

## Optimized function to get the fixture dictionary

# --- Data Loading and Processing ---
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]
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
# Map each team's logo image to the league DataFrame
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)
# Calculate and assign rankings in the league DataFramae


league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

########################################
def get_home_away_str_dict(new_fdr_df, new_fixt_df):
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    
    for col in new_fdr_df.columns:
        for value, string in zip(new_fdr_df[col], new_fixt_df[col]):
            if value not in result_dict:
                result_dict[value] = set()
            result_dict[value].add(string)

    for key in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
        result_dict.setdefault(key, [])
    return result_dict

home_away_dict = get_home_away_str_dict()

def color_fixtures(val):
    if val in home_away_dict[1]:
        return '#147d1b'
    elif val in home_away_dict[1.5]:
        return '#0ABE4A'
    elif val in home_away_dict[2]:
        return '#00ff78'
    elif val in home_away_dict[2.5]:
        return "#caf4bd"
    elif val in home_away_dict[3]:
        return '#eceae6'
    elif val in home_away_dict[3.5]:
        return "#fa8072"
    elif val in home_away_dict[4]:
        return '#ff0057'
    elif val in home_away_dict[4.5]:
        return '#C9054F'
    elif val in home_away_dict[5]:
        return '#920947'
    else:
        return 'none'  # No background color


#########################################
# --- Streamlit App ---
st.title("Premier League Table")

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

# --- Column Definitions ---
col_defs = [
    ColumnDefinition(name="Rank", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="logo_team", textprops={'ha': "center", 'va': "center"}, plot_fn=image, width=1),
    ColumnDefinition(name="Team", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="GP", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="W", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="D", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="L", group="Matches Played", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="GF", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="GA", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="GD", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="CS", group="Goals", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="Pts", group="Points", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="Pts/Game", group="Points", textprops={'ha': "center"}, width=0.5),
    ColumnDefinition(name="Form", group="Points", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="GF/Game", group="ByGame", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="GA/Game", group="ByGame", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name="CS/Game", group="ByGame", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name=f"GW{ct_gw}", group="Fixtures", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name=f"GW{ct_gw+1}", group="Fixtures", textprops={'ha': "center"}, width=1),
    ColumnDefinition(name=f"GW{ct_gw+2}", group="Fixtures", textprops={'ha': "center"}, width=1)
]
# --- Plottable Table ---
fig, ax = plt.subplots(figsize=(16, 10))  # Adjust figsize for Streamlit
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

table = Table(
    league_df,
    column_definitions=col_defs,
    columns=['logo_team','Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'CS', 'Pts', 
             'Pts/Game','Form', 'GF/Game', 'GA/Game', 'CS/Game', f'GW{ct_gw}', f'GW{ct_gw+1}', f'GW{ct_gw+2}'], 
    index_col="Rank",
    row_dividers=True,
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    footer_divider=True,
    textprops={"fontsize": 14},  # Adjust fontsize for Streamlit
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": .5, "linestyle": "-"},
    ax=ax
)
for i, col in enumerate([f'GW{ct_gw}', f'GW{ct_gw+1}', f'GW{ct_gw+2}']):
    for j in range(len(league_df)):
        color = color_fixtures(league_df[col].iloc[j])
        if color != 'none':
            ax.table.cell[j + 1, i + len(col_defs) - 3].set_facecolor(color)  # Adjust index for column position
# --- Display the Table in Streamlit ---
st.pyplot(fig)