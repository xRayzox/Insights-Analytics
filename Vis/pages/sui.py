import streamlit as st
import datetime as dt
import altair as alt
import pandas as pd
import requests
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.patches as patches 
from PIL import Image
from urllib.request import urlopen
import random
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from PIL import Image, ImageDraw, ImageOps
from PIL import Image
import textwrap
from urllib.request import urlopen
import io
import requests
from functools import lru_cache
from io import BytesIO





pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data, get_manager_history_data, get_manager_team_data,
    get_manager_details, get_player_data, get_current_season
)
from fpl_utils import (
    define_sidebar, chip_converter,get_total_fpl_players
)
from fpl_league import (
    fetch_league_info,
    get_manager_details,
    get_manager_history_data,
    get_bootstrap_data,

)

from fpl_params import MY_FPL_ID, BASE_URL
st.set_page_config(page_title='Manager', page_icon=':necktie:', layout='wide')
st.markdown(
    """
    <style>
    body {
        background-color: #181818;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with open('./data/wave.css') as f:
        css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
########################################################

define_sidebar()
st.title('Manager')

import duckdb
import pandas as pd
import time
import streamlit as st

start_duckdb = time.time()

# Cache the data loading function
@st.cache_data
def load_manager_data():
    start_duckdb = time.time()

    # Load all CSVs using DuckDB
    history_manager_duckdb = duckdb.query(""" 
        SELECT * FROM read_csv_auto('./data/manager/clean_Managers_part*.csv')
    """).to_df()

    end_duckdb = time.time()
    print("DuckDB Runtime:", end_duckdb - start_duckdb)

    return history_manager_duckdb

# Call the cached function to load the data
history_manager_pandas = load_manager_data()

# Optional: Convert to Pandas if needed
history_manager_pandas = pd.DataFrame(history_manager_pandas)
end_duckdb = time.time()
print("DuckDB Runtime:", end_duckdb - start_duckdb)


test=history_manager_pandas.head(20)



selected_teams = st.multiselect(
                label='Show teams',
                options=test['Manager'],
                default="Wael Hcine",
                format_func=lambda x: x,
            )