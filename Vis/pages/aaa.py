import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plottable import ColumnDefinition, Table
import urllib.request
import sys
import os

import streamlit as st
# Assuming you have already defined the functions to fetch data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
# from your FPL API collection
from fpl_api_collection import get_league_table, get_current_gw, get_fixt_dfs

# Define your color map for the fixtures
fixture_cmap = LinearSegmentedColormap.from_list(
    name="FixtureColorMap", colors=["#147d1b", "#0ABE4A", "#00ff78", "#caf4bd", "#eceae6", "#fa8072", "#ff0057", "#C9054F", "#920947"], N=256
)

# Load your league data
league_df = get_league_table()
ct_gw = get_current_gw()
team_fixt_df = get_fixt_dfs()  # This should contain your fixture data

# Prepare the DataFrame for the fixtures
fixture_data = []
for gw in range(ct_gw, ct_gw + 3):
    fixture_col = team_fixt_df[str(gw)].tolist()  # Adjust as per your fixture DataFrame structure
    fixture_data.append(fixture_col)

# Create a new DataFrame for display
fixture_df = pd.DataFrame(fixture_data, columns=league_df['Team'].tolist(), index=[f'GW{gw}' for gw in range(ct_gw, ct_gw + 3)])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 5))

# Column definitions including the fixture columns
column_definitions = [
    ColumnDefinition(name, cmap=fixture_cmap, formatter=lambda x: "") for name in fixture_df.columns
] + [ColumnDefinition("Index", title="", width=1.5, textprops={"ha": "right"})]

# Create the table
tab = Table(
    fixture_df,
    column_definitions=column_definitions,
    row_dividers=False,
    col_label_divider=False,
    textprops={"ha": "center", "fontname": "Roboto"},
    cell_kw={
        "edgecolor": "w",
        "linewidth": 0,
    },
)

tab.col_label_row.set_facecolor("k")
tab.col_label_row.set_fontcolor("w")
tab.columns["Index"].set_facecolor("k")
tab.columns["Index"].set_fontcolor("w")
tab.columns["Index"].set_linewidth(0)

# Show the figure
plt.show()

# Save the figure
fig.savefig("images/premier_league_fixtures.png", dpi=200)

st.plotly_chart(fig)