import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Interactive Soccer Pitch", layout="wide")

# Title
st.title("Interactive Soccer Pitch")

# Create a dataframe to store player positions
if 'players' not in st.session_state:
    st.session_state.players = pd.DataFrame(columns=['x', 'y', 'team'])

# Function to create the pitch
def create_pitch():
    pitch_layout = go.Layout(
        xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-5, 75], showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        width=800,
        height=600,
        plot_bgcolor='#3CB371'  # Green color for the pitch
    )

    pitch = go.Figure(layout=pitch_layout)

    # Add pitch lines
    pitch.add_shape(type="rect", x0=0, y0=0, x1=100, y1=70, line=dict(color="White"))
    pitch.add_shape(type="rect", x0=0, y0=20, x1=16.5, y1=50, line=dict(color="White"))
    pitch.add_shape(type="rect", x0=83.5, y0=20, x1=100, y1=50, line=dict(color="White"))
    pitch.add_shape(type="circle", x0=48, y0=33, x1=52, y1=37, line=dict(color="White"))

    # Add center line
    pitch.add_shape(type="line", x0=50, y0=0, x1=50, y1=70, line=dict(color="White"))

    return pitch

# Sidebar for adding players
st.sidebar.header("Add Player")
x_pos = st.sidebar.slider("X Position", 0, 100, 50)
y_pos = st.sidebar.slider("Y Position", 0, 70, 35)
team = st.sidebar.selectbox("Team", ["Home", "Away"])

if st.sidebar.button("Add Player"):
    new_player = pd.DataFrame({'x': [x_pos], 'y': [y_pos], 'team': [team]})
    st.session_state.players = pd.concat([st.session_state.players, new_player], ignore_index=True)

# Create pitch
pitch = create_pitch()

# Add players to the pitch
for _, player in st.session_state.players.iterrows():
    color = 'red' if player['team'] == 'Home' else 'blue'
    pitch.add_trace(go.Scatter(x=[player['x']], y=[player['y']], mode='markers', 
                               marker=dict(size=10, color=color)))

# Display the pitch
st.plotly_chart(pitch, use_container_width=True)

# Display player data
st.subheader("Player Positions")
st.dataframe(st.session_state.players)

# Button to clear all players
if st.button("Clear All Players"):
    st.session_state.players = pd.DataFrame(columns=['x', 'y', 'team'])
    st.experimental_rerun()