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





AgGrid(league_df, height=400)