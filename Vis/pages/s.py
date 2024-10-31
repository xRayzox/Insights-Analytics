import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
import sys
import os
from st_aggrid import AgGrid, GridOptionsBuilder

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

# Adding color for home and away games
def color_fixtures(val):
    color_map = {
        1: "#147d1b",
        1.5: "#0ABE4A",
        2: "#00ff78",
        2.5: "#caf4bd",
        3: "#eceae6",
        3.5: "#fa8072",
        4: "#ff0057",
        4.5: "#C9054F",
        5: "#920947",
    }
    return color_map.get(val, "#FF0000")  # Default color if no match


# --- Streamlit App ---
st.title("Premier League Table")

# Convert the 'league_df' into a DataFrame ready for AG Grid
league_df['Form'] = league_df['Form'].apply(lambda x: ''.join(form_color(x)))  # Apply form colors as a string

# --- Configuring AG Grid ---
gb = GridOptionsBuilder.from_dataframe(league_df)
gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb.configure_column("Rank", sort='asc')
gridOptions = gb.build()

# --- Display the Table in Streamlit with AG Grid ---
AgGrid(
    league_df,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    allow_unsafe_jscode=True,  # This is required to render the colored cells properly
    theme='alpine'  # Select the theme
)
