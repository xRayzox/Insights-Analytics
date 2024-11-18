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
st.title("Welcome to FPL Insights")

# Introduction Section
st.header("Unlock Your Fantasy Premier League Potential")
st.write("""
FPL Insights is your ultimate tool for data-driven decision-making in Fantasy Premier League. 
With detailed performance analysis, fixture insights, and customized recommendations, 
this app helps you make informed choices to optimize your FPL strategy throughout the season.
""")

# Features Section
st.header("Features")

# Feature List with Icons and Descriptions
st.markdown("""
- **Player Performance Analysis**:  
  Dive deep into player stats such as goals, assists, and minutes played, to identify top performers and optimize transfers.
  
- **Fixture Difficulty Rating (FDR)**:  
  Visualize upcoming fixture difficulty to plan your transfers, captain picks, and team strategy effectively.
  
- **Gameweek Trends**:  
  Analyze player and team performance trends across gameweeks, spotting rising stars and underperforming players.

- **Customized Recommendations**:  
  Get tailored data-driven insights for transfers, captain choices, and optimal squad selection.

- **Team Performance & Comparison**:  
  Track team standings with key metrics, and compare teams across various parameters for better decision-making.

- **Manager Dashboard**:  
  View detailed team history, including past gameweek performance, season stats, and league rankings with interactive charts.

- **Player Comparison**:  
  Compare players' gameweek performance and advanced metrics, visualized through comparative charts for easy analysis.

- **Fixture & FDR Insights**:  
  Access a comprehensive list of fixtures and results, with an advanced FDR matrix to evaluate matchups between teams.

- **Optimal Team Selection**:  
  Predict the best lineup for the next gameweek using machine learning (XGBoost), optimizing your strategy based on performance data.
""")

# Call to Action
st.header("Get Started")
st.write("""
Start exploring the insights and take your Fantasy Premier League strategy to the next level. 
Choose a section from the sidebar to begin analyzing player data, tracking fixture difficulty, or optimizing your team selection.
""")

# Footer Section
st.write("Made with ❤️ by FPL Insights")








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
        time.sleep(1200)  # 1200 seconds = 20 minutes


keep_streamlit_awake()