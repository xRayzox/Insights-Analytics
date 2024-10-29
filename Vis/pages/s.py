import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
import sys
import os
import reactable as rt

pd.set_option('future.no_silent_downcasting', True)

# Adjust the path to include the FPL directory 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))

# Now you can import your modules
from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)

# --- Streamlit Configuration ---
st.set_page_config(layout="wide")

# --- Functions ---
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image = Image.open(response).convert("RGBA")
    return image

def style_form_string(value):
    color_mapping = {
        'W': '#28a745',
        'D': '#ffc107',
        'L': '#dc3545',
    }
    styled_chars = [
        f"<span style='padding: 2px; border-radius: 3px; background-color: {color_mapping.get(char, '#FFFFFF')}'>{char}</span>" 
        for char in value
    ]
    return ''.join(styled_chars)

# --- Data Loading and Processing ---
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]
league_df = league_df.join(new_fixt_df)
league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + "@x2.png"
teams_df['logo_image'] = teams_df['logo_url'].apply(load_image_from_url)
team_logo_mapping = pd.Series(teams_df['logo_image'].values, index=teams_df['short_name']).to_dict()
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---
st.title("Premier League Table")

# --- Reactable Table ---
st.write(
    rt.reactable(
        league_df,
        columns={
            "Rank": rt.colDef(name="Rank", width=50),
            "logo_team": rt.colDef(
                name="Team Logo",
                cell=rt.Image(src='logo_team'),
                width=70
            ),
            "Team": rt.colDef(name="Team", width=150),
            "GP": rt.colDef(name="GP", group="Matches Played", width=50),
            "W": rt.colDef(name="W", group="Matches Played", width=50),
            "D": rt.colDef(name="D", group="Matches Played", width=50),
            "L": rt.colDef(name="L", group="Matches Played", width=50),
            "GF": rt.colDef(name="GF", group="Goals", width=50),
            "GA": rt.colDef(name="GA", group="Goals", width=50),
            "GD": rt.colDef(name="GD", group="Goals", width=50),
            "CS": rt.colDef(name="CS", group="Goals", width=50),
            "Pts": rt.colDef(name="Pts", group="Points", width=50),
            "Pts/Game": rt.colDef(name="Pts/Game", group="Points", width=80),
            "Form": rt.colDef(
                name="Form",
                group="Points",
                cell=lambda value: rt.HTML(style_form_string(value)),
                width=80
            ),
            "GF/Game": rt.colDef(name="GF/Game", group="By Game", width=80),
            "GA/Game": rt.colDef(name="GA/Game", group="By Game", width=80),
            "CS/Game": rt.colDef(name="CS/Game", group="By Game", width=80),
            f"GW{ct_gw}": rt.colDef(name=f"GW{ct_gw}", group="Fixtures", width=80),
            f"GW{ct_gw + 1}": rt.colDef(name=f"GW{ct_gw + 1}", group="Fixtures", width=80),
            f"GW{ct_gw + 2}": rt.colDef(name=f"GW{ct_gw + 2}", group="Fixtures", width=80)
        },
        default_col_size=50,
        pagination=False,
        show_grid=True,
        row_style={
            "border-bottom": "1px solid #eee",
        },
        theme={
            "row_highlight_background_color": "#f0f5f9"
        }
    ),
)
