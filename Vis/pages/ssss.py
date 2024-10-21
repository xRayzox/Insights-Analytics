import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    plt.plot(,, marker="o", markersize=5, color="white")

    # Set the pitch color
    ax.set_facecolor('#3CB371')

    # Remove axis labels
    plt.axis('off')

def add_player(ax, x, y, number, name, role, team):
    color = 'red' if team == 'Home' else 'blue'
    circle = plt.Circle((x, y), 2, color=color)
    ax.add_artist(circle)
    plt.text(x, y-3, number, ha='center', va='center', color='white', fontweight='bold')
    plt.text(x, y+3, name, ha='center', va='center', color='white', fontsize=8)

    # Add tooltip
    tooltip = patches.Rectangle((x-5, y-5), 10, 10, fill=False, edgecolor='white', linewidth=1, alpha=0)
    tooltip.set_visible(False)
    ax.add_patch(tooltip)
    tooltip.set_label(f"{number} - {name} ({role})")
    tooltip.set_picker(True)

    def on_pick(event):
        if event.artist == tooltip:
            tooltip.set_visible(not tooltip.get_visible())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

st.title("4-3-3 Soccer Formation with Tooltips")

# Input player data
home_players = {
    'GK': {'name': 'John Doe', 'x': 5, 'y': 45},
    'RB': {'name': 'Jane Smith', 'x': 20, 'y': 70},
    'CB': {'name': 'Alice Brown', 'x': 40, 'y': 70},
    'CB': {'name': 'Bob Johnson', 'x': 60, 'y': 70},
    'LB': {'name': 'Charlie Davis', 'x': 80, 'y': 70},
    'DM': {'name': 'Eve Wilson', 'x': 50, 'y': 55},
    'CM': {'name': 'Frank Lee', 'x': 35, 'y': 45},
    'CM': {'name': 'Grace Martin', 'x': 65, 'y': 45},
    'RW': {'name': 'Hank Nelson', 'x': 20, 'y': 20},
    'LW': {'name': 'Ivan Thompson', 'x': 80, 'y': 20},
    'ST': {'name': 'Julia White', 'x': 50, 'y': 20}
}

away_players = {
    'GK': {'name': 'Michael Black', 'x': 95, 'y': 45},
    'RB': {'name': 'Nancy Green', 'x': 80, 'y': 70},
    'CB': {'name': 'Oliver Brown', 'x': 60, 'y': 70},
    'CB': {'name': 'Patricia Johnson', 'x': 40, 'y': 70},
    'LB': {'name': 'Quentin Davis', 'x': 20, 'y': 70},
    'DM': {'name': 'Rachel Wilson', 'x': 50, 'y': 55},
    'CM': {'name': 'Samuel Lee', 'x': 65, 'y': 45},
    'CM': {'name': 'Tina Martin', 'x': 35, 'y': 45},
    'RW': {'name': 'Ursula Nelson', 'x': 80, 'y': 20},
    'LW': {'name': 'Victor Thompson', 'x': 20, 'y': 20},
    'ST': {'name': 'Wendy White', 'x': 50, 'y': 20}
}

if st.button("Generate Lineup"):
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # Add home team players
    for role, player in home_players.items():
        add_player(ax, player['x'], player['y'], "", player['name'], role, 'Home')

    # Add away team players
    for role, player in away_players.items():
        add_player(ax, player['x'], player['y'], "", player['name'], role, 'Away')

    st.pyplot(fig)