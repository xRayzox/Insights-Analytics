import streamlit as st
import sys
import os
import sys
import pandas as pd
import time
import requests
pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FPL')))

# Set the path to the 'fpl_utils' directory
from fpl_utils import (
    define_sidebar, chip_converter
)
st.set_page_config(page_title='Home', page_icon=':house:', layout='wide')


with open('./data/wave.css') as f:
        css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Project Description
define_sidebar()
st.title("Fantasy Premier League Analysis and Insights ")
st.markdown("""
### Objective:
This application provides a detailed analysis of Fantasy Premier League (FPL) data, offering key insights to help users make better decisions throughout the season.

### Features:
- **Player Performance Analysis**: Detailed breakdown of player stats such as goals, assists, and minutes played, helping users identify top performers.
- **Fixture Difficulty Rating (FDR)**: Visual representation of team fixture difficulties, enabling users to plan transfers and captain choices based on future fixtures.
- **Gameweek Trends**: Analysis of team and player trends across gameweeks, allowing for early identification of rising stars or underperforming players.
- **Customized Recommendations**: Data-driven insights to assist in making transfers, picking captains, and optimizing squad selection.

This tool leverages various FPL datasets to generate interactive visualizations, making it easier to analyze performance and optimize your fantasy football strategy.
""")


def keep_streamlit_awake():
    while True:
        try:
            # Replace with your Streamlit app's URL
            response = requests.get("https://fpl-insights.streamlit.app")
            if response.status_code == 200:
                print("Streamlit app is awake!")
            else:
                print(f"Error: Received status code {response.status_code}")
        except Exception as e:
            print(f"Error pinging Streamlit app: {e}")
        
        # Wait for 20 minutes before sending the next request
        time.sleep(1)  # 1200 seconds = 20 minutes


keep_streamlit_awake()