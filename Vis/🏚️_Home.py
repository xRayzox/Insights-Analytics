import streamlit as st
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FPL')))

# Set the path to the 'fpl_utils' directory
from fpl_utils import (
    define_sidebar, chip_converter
)
st.set_page_config(page_title='Home', page_icon=':house:', layout='wide')
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