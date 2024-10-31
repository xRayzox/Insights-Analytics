import streamlit as st
import pandas as pd

def calculate_standing(matches):
    """Calculates the league standing based on match results."""

    teams = set()
    for match in matches:
        teams.add(match["home_team"])
        teams.add(match["away_team"])

    standings = {team: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "goals_for": 0, "goals_against": 0, "points": 0} for team in teams}


    for match in matches:
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]

        standings[home_team]["played"] += 1
        standings[away_team]["played"] += 1

        standings[home_team]["goals_for"] += home_score
        standings[home_team]["goals_against"] += away_score
        standings[away_team]["goals_for"] += away_score
        standings[away_team]["goals_against"] += home_score


        if home_score > away_score:
            standings[home_team]["won"] += 1
            standings[away_team]["lost"] += 1
            standings[home_team]["points"] += 3
        elif home_score < away_score:
            standings[away_team]["won"] += 1
            standings[home_team]["lost"] += 1
            standings[away_team]["points"] += 3
        else:
            standings[home_team]["drawn"] += 1
            standings[away_team]["drawn"] += 1
            standings[home_team]["points"] += 1
            standings[away_team]["points"] += 1

    return standings



def display_standing(standings):
    """Displays the league standing in a Streamlit table."""

    standing_list = []
    for team, stats in standings.items():
        stats["team"] = team  # Add team name to the stats dictionary
        standing_list.append(stats)


    df = pd.DataFrame(standing_list)
    df = df.sort_values(by=["points", "goals_for", "goals_against"], ascending=[False, False, True])
    df = df[["team", "played", "won", "drawn", "lost", "goals_for", "goals_against",  "points"]] # Reorder columns
    st.dataframe(df)


st.title("Football League Standing")

# Sample match data (you can replace this with user input or data from a file)
matches = [
    {"home_team": "Team A", "away_team": "Team B", "home_score": 2, "away_score": 1},
    {"home_team": "Team C", "away_team": "Team D", "home_score": 0, "away_score": 0},
    {"home_team": "Team B", "away_team": "Team C", "home_score": 3, "away_score": 2},
     {"home_team": "Team A", "away_team": "Team D", "home_score": 1, "away_score": 1},
    # Add more matches...
]

# Example using a form to add matches
with st.form("add_match_form"):
    st.write("Add Match Result:")
    home_team = st.text_input("Home Team")
    away_team = st.text_input("Away Team")
    home_score = st.number_input("Home Score", min_value=0, step=1)
    away_score = st.number_input("Away Score", min_value=0, step=1)
    submitted = st.form_submit_button("Add Match")

    if submitted:
        matches.append({"home_team": home_team, "away_team": away_team, "home_score": home_score, "away_score": away_score})


standings = calculate_standing(matches)
display_standing(standings)