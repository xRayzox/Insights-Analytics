import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_pitch(ax):
    # Pitch outline & centre line
    plt.plot([0,0,100,100,0],[0,100,100,0,0], color="white")
    plt.plot([50,50], [0,100], color="white")

    # Left penalty area
    plt.plot([16.5,16.5], [80,20], color="white")
    plt.plot([0,16.5], [80,80], color="white")
    plt.plot([0,16.5], [20,20], color="white")

    # Right penalty area
    plt.plot([83.5,83.5], [80,20], color="white")
    plt.plot([100,83.5], [80,80], color="white")
    plt.plot([100,83.5], [20,20], color="white")

    # Left 6-yard box
    plt.plot([0,5.5], [65,65], color="white")
    plt.plot([5.5,5.5], [65,35], color="white")
    plt.plot([0,5.5], [35,35], color="white")

    # Right 6-yard box
    plt.plot([100,94.5], [65,65], color="white")
    plt.plot([94.5,94.5], [65,35], color="white")
    plt.plot([100,94.5], [35,35], color="white")

    # Centre circle
    circle = plt.Circle((50, 50), 9.15, fill=False, color="white")
    ax.add_artist(circle)

    # Spot the ball
    plt.plot([50], [50], marker="o", markersize=5, color="white")

    # Set the pitch color
    ax.set_facecolor('#3CB371')

    # Remove axis labels
    plt.axis('off')

def add_player(ax, x, y, number, name, team):
    color = 'red' if team == 'Home' else 'blue'
    circle = plt.Circle((x, y), 2, color=color)
    ax.add_artist(circle)
    plt.text(x, y-3, number, ha='center', va='center', color='white', fontweight='bold')
    plt.text(x, y+3, name, ha='center', va='center', color='white', fontsize=8)

st.title("Football/Soccer Match Lineup")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Home Team")
    home_players = {}
    for i in range(11):
        name = st.text_input(f"Home Player {i+1} Name", key=f"home_name_{i}")
        x = st.number_input(f"X position for {name}", min_value=0, max_value=100, value=50, key=f"home_x_{i}")
        y = st.number_input(f"Y position for {name}", min_value=0, max_value=100, value=50, key=f"home_y_{i}")
        home_players[i+1] = {"name": name, "x": x, "y": y}

with col2:
    st.subheader("Away Team")
    away_players = {}
    for i in range(11):
        name = st.text_input(f"Away Player {i+1} Name", key=f"away_name_{i}")
        x = st.number_input(f"X position for {name}", min_value=0, max_value=100, value=50, key=f"away_x_{i}")
        y = st.number_input(f"Y position for {name}", min_value=0, max_value=100, value=50, key=f"away_y_{i}")
        away_players[i+1] = {"name": name, "x": x, "y": y}

if st.button("Generate Lineup"):
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    for number, player in home_players.items():
        add_player(ax, player['x'], player['y'], number, player['name'], 'Home')

    for number, player in away_players.items():
        add_player(ax, player['x'], player['y'], number, player['name'], 'Away')

    st.pyplot(fig)
