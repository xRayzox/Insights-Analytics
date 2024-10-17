import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime, timezone

# Adjust the path to your FPL API collection as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_fixture_data
)
from fpl_utils import (
    define_sidebar,
    get_annot_size,
    map_float_to_color,
    get_text_color_from_hash,
    get_rotation,
    get_user_timezone
)

# Streamlit app
st.set_page_config(page_title="FPL Fixture Analysis", layout="wide")
st.title("FPL Fixture Analysis")

# Load data outside the conditional block
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
events_df = pd.DataFrame(get_bootstrap_data()['events'])
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])

# Get current gameweek
ct_gw = get_current_gw()

# --- Data Preprocessing ---
# Prepare dataframes for FDR, GA, GF
def prepare_data(df):
    df = df.reset_index()
    df.rename(columns={0: 'Team'}, inplace=True)
    df = df.melt(id_vars='Team', var_name='GameWeek', value_name=df.columns[1])
    df['GameWeek'] = df['GameWeek'].astype(int)
    return df

fdr_matrix = prepare_data(team_fdr_df.copy())
ga_matrix = prepare_data(team_ga_df.copy())
gf_matrix = prepare_data(team_gf_df.copy())

# Round GA and GF values to 2 decimal places
ga_matrix['GA'] = ga_matrix['GA'].astype(float).round(2)
gf_matrix['GF'] = gf_matrix['GF'].astype(float).round(2)

# --- Functions for Styling and Display ---
def color_fdr(value):
    fdr_colors = {
        1: ("#257d5a", "black"),
        2: ("#00ff86", "black"),
        3: ("#ebebe4", "black"),
        4: ("#ff005a", "white"),
        5: ("#861d46", "white"),
    }
    if value in fdr_colors:
        background_color, text_color = fdr_colors[value]
        return f'background-color: {background_color}; color: {text_color}; text-align: center;'
    else:
        return ''

def color_ga_gf(value):
    ga_gf_colors = {
        0.0: ("#147d1b", "white"),
        0.5: ("#00ff78", "black"),
        1.0: ("#caf4bd", "black"),
        1.5: ("#eceae6", "black"),
        2.0: ("#fa8072", "black"),
        2.5: ("#ff0057", "white"),
        3.0: ("#920947", "white"), 
    }
    rounded_value = round(value, 1)  # Round to one decimal place for display and color mapping
    closest_key = min(ga_gf_colors, key=lambda x: abs(x - rounded_value))
    background_color, text_color = ga_gf_colors[closest_key]
    return f'background-color: {background_color}; color: {text_color}; text-align: center;'


def display_metric_table(selected_metric, df, start_gw, end_gw):
    filtered_df = df[(df['GameWeek'] >= start_gw) & (df['GameWeek'] <= end_gw)]
    pivot_df = filtered_df.pivot(index='Team', columns='GameWeek', values=selected_metric)
    pivot_df.columns = [f'GW {col}' for col in pivot_df.columns]

    # Apply styling based on metric
    if selected_metric == "FDR":
        styled_table = pivot_df.style.applymap(color_fdr)
    else:
        styled_table = pivot_df.style.applymap(color_ga_gf)

    st.markdown(
        f"**{selected_metric} for the Next {end_gw - start_gw + 1} Gameweeks (Starting GW {start_gw})**",
        unsafe_allow_html=True
    )
    st.write(styled_table)

    # Display Legend
    with st.sidebar:
        st.markdown(f"**Legend ({selected_metric}):**")
        if selected_metric == "FDR":
            for fdr, (bg_color, font_color) in {
                1: ("#257d5a", "black"),
                2: ("#00ff86", "black"),
                3: ("#ebebe4", "black"),
                4: ("#ff005a", "white"),
                5: ("#861d46", "white"),
            }.items():
                st.sidebar.markdown(
                    f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
                    f"{fdr} - {'Very Easy' if fdr == 1 else 'Easy' if fdr == 2 else 'Medium' if fdr == 3 else 'Difficult' if fdr == 4 else 'Very Difficult'}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
        else:
            for ga_gf, (bg_color, font_color) in {
                0.0: ("#147d1b", "white"),
                0.5: ("#00ff78", "black"),
                1.0: ("#caf4bd", "black"),
                1.5: ("#eceae6", "black"),
                2.0: ("#fa8072", "black"),
                2.5: ("#ff0057", "white"),
                3.0: ("#920947", "white"),
            }.items():
                st.sidebar.markdown(
                    f"<span style='background-color: {bg_color}; color: {font_color}; padding: 2px 5px; border-radius: 3px;'>"
                    f"{ga_gf:.1f} - {ga_gf + 0.4:.1f}"
                    f"</span>",
                    unsafe_allow_html=True,
                )


# --- Fixture Display Functions ---
def display_fixtures(selected_gw):
    fixtures_df = pd.DataFrame(get_fixture_data())
    fixtures_df.drop(columns='stats', inplace=True)
    fixtures_df = fixtures_df.merge(teams_df[['id', 'name', 'logo_url']], left_on='team_h', right_on='id', how='left').rename(
        columns={'name': 'team_h_name', 'logo_url': 'team_h_logo'}
    )
    fixtures_df = fixtures_df.merge(teams_df[['id', 'name', 'logo_url']], left_on='team_a', right_on='id', how='left').rename(
        columns={'name': 'team_a_name', 'logo_url': 'team_a_logo'}
    )
    fixtures_df = fixtures_df.drop(columns=['pulse_id'])
    fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
    fixtures_df['local_time'] = fixtures_df['datetime'].dt.tz_convert(get_user_timezone()).dt.strftime('%A %d %B %Y %H:%M')
    fixtures_df['local_date'] = fixtures_df['datetime'].dt.tz_convert(get_user_timezone()).dt.strftime('%d %A %B %Y')
    fixtures_df['local_hour'] = fixtures_df['datetime'].dt.tz_convert(get_user_timezone()).dt.strftime('%H:%M')

    st.markdown(
        f"<h2 style='text-align: center;'>Premier League Fixtures - Gameweek {selected_gw}</h2>",
        unsafe_allow_html=True,
    )

    current_gameweek_fixtures = fixtures_df[fixtures_df['event'] == selected_gw]
    grouped_fixtures = current_gameweek_fixtures.groupby('local_date')

    with st.container():
        for date, matches in grouped_fixtures:
            st.markdown(f"<h3 style='text-align: center;'>{date}</h3>", unsafe_allow_html=True)
            for _, match in matches.iterrows():
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown(
                        f"<div style='text-align: right;'>"
                        f"{match['team_h_name']} "
                        f"<img src='{match['team_h_logo']}' style='width:20px; height:20px; vertical-align:middle; margin-left:5px;'/></div>",
                        unsafe_allow_html=True
                    )

                with col2:
                    if match['finished']:
                        st.markdown(
                            f"<div style='text-align: center;'>{int(match['team_h_score'])} - {int(match['team_a_score'])}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div style='text-align: center;'>vs</div>",
                            unsafe_allow_html=True
                        )

                with col3:
                    st.markdown(
                        f"<div style='text-align: left;'>"
                        f"<img src='{match['team_a_logo']}' style='width:20px; height:20px; vertical-align:middle; margin-right:5px;'/>"
                        f"{match['team_a_name']}</div>",
                        unsafe_allow_html=True
                    )

                    if not match['finished']:
                        st.markdown(
                            f"<p style='text-align: center; margin-top: 10px;'>Kickoff: {match['local_hour']}</p>",
                            unsafe_allow_html=True
                        )


# --- Sidebar and Main App Logic ---
with st.sidebar:
    selected_display = st.radio(
        "Select Display:", ['⚔️ Premier League Fixtures', '📊 Fixture Data']
    )

if selected_display == '📊 Fixture Data':
    gw_min = min(events_df['id'])
    gw_max = max(events_df['id'])
    start_gw, end_gw = st.slider('Gameweek Range:', gw_min, gw_max, [ct_gw, ct_gw + 10], 1)

    selected_metric = st.selectbox(
        "Select Metric:",
        ("FDR", "GA", "GF")
    )

    if selected_metric == "FDR":
        display_metric_table("FDR", fdr_matrix, start_gw, end_gw)
    elif selected_metric == "GA":
        display_metric_table("GA", ga_matrix, start_gw, end_gw)
    elif selected_metric == "GF":
        display_metric_table("GF", gf_matrix, start_gw, end_gw)

elif selected_display == '⚔️ Premier League Fixtures':
    gw_minn = min(pd.DataFrame(get_fixture_data())['event'])
    gw_maxx = max(pd.DataFrame(get_fixture_data())['event'])
    selected_gw = st.slider('Select Gameweek:', gw_minn, gw_maxx, ct_gw)
    display_fixtures(selected_gw)

define_sidebar()