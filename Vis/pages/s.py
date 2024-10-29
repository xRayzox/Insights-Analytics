import streamlit as st
from pathlib import Path
import pandas as pd
import urllib.request
import os
import sys
from PIL import Image  # Importing Image from PIL
from bokeh.models import ColumnDataSource, TableColumn, DataTable, HTMLTemplateFormatter

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
source = ColumnDataSource(data=dict(
    Rank=[row[0] for row in table_data],
    Logo=[row[1] for row in table_data],
    Team=[row[2] for row in table_data],
    GP=[row[3] for row in table_data],
    W=[row[4] for row in table_data],
    D=[row[5] for row in table_data],
    L=[row[6] for row in table_data],
    GF=[row[7] for row in table_data],
    GA=[row[8] for row in table_data],
    GD=[row[9] for row in table_data],
    CS=[row[10] for row in table_data],
    Pts=[row[11] for row in table_data],
    Pts_Game=[row[12] for row in table_data],
    Form=[row[13] for row in table_data],
    GF_Game=[row[14] for row in table_data],
    GA_Game=[row[15] for row in table_data],
    CS_Game=[row[16] for row in table_data],
    GW_Current=[row[17] for row in table_data],
    GW_Next1=[row[18] for row in table_data],
    GW_Next2=[row[19] for row in table_data],
))

# Define the table columns
table_columns = [
    TableColumn(field='Rank', title='Rank'),
    TableColumn(field='Logo', title='Logo', formatter=HTMLTemplateFormatter()),
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
    TableColumn(field='Pts_Game', title='Pts/Game'),
    TableColumn(field='Form', title='Form'),
    TableColumn(field='GF_Game', title='GF/Game'),
    TableColumn(field='GA_Game', title='GA/Game'),
    TableColumn(field='CS_Game', title='CS/Game'),
    TableColumn(field='GW_Current', title=f'GW{ct_gw}'),
    TableColumn(field='GW_Next1', title=f'GW{ct_gw+1}'),
    TableColumn(field='GW_Next2', title=f'GW{ct_gw+2}'),
]

# Create a Bokeh DataTable
data_table = DataTable(source=source, columns=table_columns, width=1000, height=800)

# --- Display the Table in Streamlit ---
st.bokeh_chart(data_table)
