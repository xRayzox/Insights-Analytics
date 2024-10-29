import streamlit as st
from pathlib import Path
import pandas as pd
import urllib.request
import os
import sys

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

# --- Data Loading and Processing ---
league_df = get_league_table()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
ct_gw = get_current_gw()
new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]
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

# Calculate and assign rankings in the league DataFrame
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---
st.title("Premier League Table")

# --- Table Styling ---
row_colors = {
    "top4": "#E1FABC",
    "top6": "#FFFC97",
    "relegation": "#E79A9A",
}

# Prepare data for Bokeh table
table_data = []
for index, row in league_df.iterrows():
    table_data.append([
        row['Rank'],
        f'<img src="{row["logo_team"]}" width="50" height="50">',
        row['Team'],
        row['GP'],
        row['W'],
        row['D'],
        row['L'],
        row['GF'],
        row['GA'],
        row['GD'],
        row['CS'],
        row['Pts'],
        row['Pts/Game'],
        row['Form'],
        row['GF/Game'],
        row['GA/Game'],
        row['CS/Game'],
        row[f'GW{ct_gw}'],
        row[f'GW{ct_gw+1}'],
        row[f'GW{ct_gw+2}']
    ])

# Define column headers
columns = ['Rank', 'Logo', 'Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'CS', 'Pts', 
           'Pts/Game', 'Form', 'GF/Game', 'GA/Game', 'CS/Game', f'GW{ct_gw}', f'GW{ct_gw+1}', f'GW{ct_gw+2}']

# Create a Bokeh DataTable
from bokeh.models import ColumnDataSource, TableColumn
from bokeh.io import curdoc
from bokeh.layouts import column

source = ColumnDataSource(data=dict(
    columns=columns,
    data=table_data
))

# Define the table columns
table_columns = [
    TableColumn(field='Rank', title='Rank'),
    TableColumn(field='Logo', title='Logo', formatter=ImageFormatter()),
    TableColumn(field='Team', title='Team'),
    TableColumn(field='GP', title='GP'),
    TableColumn(field='W', title='W'),
    TableColumn(field='D', title='D'),
    TableColumn(field='L', title='L'),
    TableColumn(field='GF', title='GF'),
    TableColumn(field='GA', title='GA'),
    TableColumn(field='GD', title='GD'),
    TableColumn(field='CS', title='CS'),
    TableColumn(field='Pts', title='Pts'),
    TableColumn(field='Pts/Game', title='Pts/Game'),
    TableColumn(field='Form', title='Form'),
    TableColumn(field='GF/Game', title='GF/Game'),
    TableColumn(field='GA/Game', title='GA/Game'),
    TableColumn(field='CS/Game', title='CS/Game'),
    TableColumn(field=f'GW{ct_gw}', title=f'GW{ct_gw}'),
    TableColumn(field=f'GW{ct_gw + 1}', title=f'GW{ct_gw + 1}'),
    TableColumn(field=f'GW{ct_gw + 2}', title=f'GW{ct_gw + 2}'),
]

# Create a Bokeh DataTable
from bokeh.models import DataTable

data_table = DataTable(source=source, columns=table_columns, width=1000, height=800)

# --- Display the Table in Streamlit ---
st.bokeh_chart(data_table)
