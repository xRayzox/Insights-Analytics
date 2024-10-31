import streamlit as st
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
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

def form_color(form):
    color_mapping = {
        'W': '#28a745',  # Green for Win
        'D': '#ffc107',  # Orange for Draw
        'L': '#dc3545',  # Red for Loss
    }
    # Create a list of colors for the form string
    return [color_mapping[char] for char in form if char in color_mapping]


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

def get_home_away_str_dict():
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
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict

    merged_dict = {k: [] for k in [1.5, 2.5, 3.5, 4.5]}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        merged_dict[k] = list(set(v))
    for i in range(1, 6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict
home_away_dict = get_home_away_str_dict()
def color_fixtures(val):
    # You can define the logic to extract the numerical part if needed
    # For example, if `val` is like "bou (h)", we might want to focus on just the numeric part
    print(f"Coloring for fixture value: {val}")  # Debug print
    # Check if the value matches any in home_away_dict
    for key in home_away_dict.keys():
        if val in home_away_dict[key]:
            return {
                1: "#147d1b",
                1.5: "#0ABE4A",
                2: "#00ff78",
                2.5: "#caf4bd",
                3: "#eceae6",
                3.5: "#fa8072",
                4: "#ff0057",
                4.5: "#C9054F",
                5: "#920947",
            }.get(key, "#FF0000")  # Default to red if no match
    
    return "#FF0000"  # Default color if no match


# Assuming league_df is defined and populated.


# Modify cmap for Fixture Column Definitions
def fixture_cmap(val):
    return color_fixtures(val)  # Directly return the color


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
    ColumnDefinition(
        name="Rank",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="logo_team",
        textprops={'ha': "center", 'va': "center", 'color': "white"},
        plot_fn=image,
        width=1,
    ),
    ColumnDefinition(
        name="Team",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="GP",
        group="Matches Played",

        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="W",
        group="Matches Played",

        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="D",
        group="Matches Played",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="L",
        group="Matches Played",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GF",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GA",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GD",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="CS",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="Pts",
        group="Points",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="Pts/Game",
        group="Points",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="Form",
        group="Points",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="GF/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="GA/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="CS/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    
]

# Modify Fixture Column Definitions
for gw in range(ct_gw, ct_gw + 3):
    col_defs.append(
        ColumnDefinition(
            name=f"GW{gw}",
            group="Fixtures",
            textprops={'ha': "center"},
            width=1,
            cmap=lambda val: color_fixtures(val)   # Use fixture_cmap directly
         )
    )


# --- Plottable Table ---
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust figsize for Streamlit
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
# Color a specific cell (e.g., color the 'Pts' of Team A)
try:
    row_index = 0  # Change to the desired row index (0 for Team A)
    col_index = 2  # Change to the desired column index (2 for Pts)
    
    # Access the specific cell and set its color
    cell_to_color = table.cells[row_index][col_index]
    cell_to_color.set_facecolor('#ffcccb')  # Light red color
except IndexError as e:
    st.error(f"IndexError: {str(e)}. Check the structure of table.cells.")
except KeyError as e:
    st.error(f"KeyError: {str(e)}. Ensure the indices used to access cells are valid.")
for idx in range(len(league_df)):
    if league_df.iloc[idx]['Rank'] <= 4:
        table.rows[idx].set_facecolor(row_colors["top4"])
    elif league_df.iloc[idx]['Rank'] <= 6:
        table.rows[idx].set_facecolor(row_colors["top6"])
    elif league_df.iloc[idx]['Rank'] >= 18:  # Assuming relegation zone starts at 18
        table.rows[idx].set_facecolor(row_colors["relegation"])



# --- Display the Table in Streamlit ---
st.pyplot(fig)