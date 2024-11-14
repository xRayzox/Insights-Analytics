import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import matplotlib.pyplot as plt  
import os 
from concurrent.futures import ThreadPoolExecutor ,ProcessPoolExecutor,as_completed
import joblib
import time
import subprocess




cwd = os.getcwd()
# Construct the full path to the 'FPL' directory
fpl_path = os.path.join(cwd, 'FPL')
print(fpl_path)
# Add it to the system path
sys.path.append(fpl_path)
from fpl_api_collection import (
    get_bootstrap_data,
    get_current_gw,
    get_fixt_dfs,
    get_fixture_data,
    get_player_id_dict,
    get_current_season,
    get_player_data,
)
import subprocess
subprocess.run(["pip", "install", "pandas"], check=True)
subprocess.run(["python", "./Vis/pages/Prediction/model.py"])

start_time = time.time()

@st.cache_data
def get_cached_fixture_data():
    return get_fixture_data()
# Cache bootstrap data retrieval to avoid repeated API calls
@st.cache_data
def get_bootstrap_data_cached():
    return get_bootstrap_data()

# Cache player dictionary and current season/gameweek retrieval
@st.cache_data
def get_player_and_season_data():
    full_player_dict = get_player_id_dict('total_points', web_name=False)
    crnt_season = get_current_season()
    ct_gw = get_current_gw()
    return full_player_dict, crnt_season, ct_gw

# Cache fixture data processing
@st.cache_data
def process_fixture_data():
    bootstrap_data = get_bootstrap_data_cached()
    teams_df = pd.DataFrame(bootstrap_data['teams'])
    team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()

    fixture_data = get_cached_fixture_data()
    fixtures_df = pd.DataFrame(fixture_data).drop(columns='stats').replace(
        {'team_h': team_name_mapping, 'team_a': team_name_mapping}
    ).drop(columns=['pulse_id'])

    timezone = 'Europe/London'
    fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
    fixtures_df[['local_time', 'local_date', 'local_hour']] = fixtures_df['datetime'].dt.tz_convert(timezone).apply(
        lambda x: pd.Series([x.strftime('%A %d %B %Y %H:%M'), x.strftime('%d %A %B %Y'), x.strftime('%H:%M')])
    )
    
    return fixtures_df

# Cache player and element type processing
@st.cache_data
def process_player_data():
    bootstrap_data = get_bootstrap_data_cached()
    teams_df = pd.DataFrame(bootstrap_data['teams'])
    element_types_df = pd.DataFrame(bootstrap_data['element_types'])
    elements_df = pd.DataFrame(bootstrap_data['elements'])

    team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()

    ele_copy = elements_df.assign(
        element_type=lambda df: df['element_type'].map(element_types_df.set_index('id')['singular_name_short']),
        team_name=lambda df: df['team'].map(teams_df.set_index('id')['short_name']),
        full_name=lambda df: df['first_name'].str.cat(df['second_name'].str.cat(df['team_name'].apply(lambda x: f" ({x})"), sep=''), sep=' ')
    )
    
    return ele_copy, team_name_mapping,teams_df

@st.cache_data
def get_cached_player_data(player_id):
    return get_player_data(str(player_id))

@st.cache_data
def get_cached_fixt_dfs():
    return get_fixt_dfs()
# Streamlit UI Components
st.title("Fantasy Premier League Data")

# Fetching and processing data
ele_copy, team_name_mapping,teams_df = process_player_data()
fixtures_df = process_fixture_data()
full_player_dict, crnt_season, ct_gw = get_player_and_season_data()
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_cached_fixt_dfs()
elapsed_time = time.time() - start_time
st.error(f"1-Time taken by my_function: {elapsed_time} seconds")




def convert_score_to_result(df):
    df['result'] = df.apply(
        lambda row: f"{row['team_h_score']}-{row['team_a_score']}" if row['was_home'] 
                    else f"{row['team_a_score']}-{row['team_h_score']}",
        axis=1
    )

def convert_opponent_string(df):
    df['vs'] += df['was_home'].apply(lambda x: ' (A)' if x else ' (H)')
    df['Team_player'] += df['was_home'].apply(lambda x: ' (H)' if x else ' (A)')
    return df

@st.cache_data
def collate_hist_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    position = ele_copy.loc[ele_copy['full_name'] == player_name, 'element_type'].iloc[0]
    Team = ele_copy.loc[ele_copy['full_name'] == player_name, 'team_name'].iloc[0]
    p_data = get_cached_player_data(str(p_id[0]))
    p_df = pd.DataFrame(p_data['history'])
    convert_score_to_result(p_df)
    p_df.loc[p_df['result'] == '<NA>-<NA>', 'result'] = '-'
    
    # Renaming columns in a single step
    rn_dict = {'round': 'GW', 'kickoff_time': 'kickoff_time', 'opponent_team': 'vs', 
               'total_points': 'Pts', 'minutes': 'Mins', 'goals_scored': 'GS', 
               'assists': 'A', 'clean_sheets': 'CS', 'goals_conceded': 'GC', 
               'own_goals': 'OG', 'penalties_saved': 'Pen_Save', 'penalties_missed': 'Pen_Miss',
               'yellow_cards': 'YC', 'red_cards': 'RC', 'saves': 'S', 'bonus': 'B', 
               'bps': 'BPS', 'influence': 'I', 'creativity': 'C', 'threat': 'T', 
               'ict_index': 'ICT', 'value': 'Price', 'selected': 'SB', 
               'transfers_in': 'Tran_In', 'transfers_out': 'Tran_Out', 
               'expected_goals': 'xG', 'expected_assists': 'xA', 
               'expected_goal_involvements': 'xGI', 'expected_goals_conceded': 'xGC', 
               'result': 'Result'}
    p_df.rename(columns=rn_dict, inplace=True)
    
    # Set column order once after renaming
    col_order = ['GW', 'kickoff_time', 'vs', 'Result', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA',
                 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S',
                 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB',
                 'Tran_In', 'Tran_Out', 'was_home']
    p_df = p_df[col_order]
    
    # Apply mappings outside loop to save time
    p_df['Price'] = p_df['Price'] / 10
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    p_df['Pos'] = position
    p_df['Team_player'] = Team
    
    # Finalize DataFrame format and return
    p_df.sort_values('GW', ascending=False, inplace=True)
    return p_df

@st.cache_data
def collate_all_players_parallel(full_player_dict, max_workers=None):
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2)  # Tune based on system capabilities

    def get_player_data_wrapper(player_name):
        try:
            player_df = collate_hist_df_from_name(player_name)
            player_df['Player'] = player_name
            return player_df
        except Exception as e:
            print(f"Error processing {player_name}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_player_data_wrapper, name): name 
                   for name in full_player_dict.values()}
        
        # Use list comprehension for faster result collection
        results = [future.result() for future in as_completed(futures) if not future.exception()]

    return pd.concat(results, axis=0, ignore_index=True)

# Run the optimized function
all_players_data = collate_all_players_parallel(full_player_dict)


elapsed_time2 = time.time() - start_time
st.error(f"2-Time taken by my_function: {elapsed_time2} seconds")
# Select only the necessary columns from teams_df for both home and away teams
team_columns = ['short_name',
                'strength_overall_home', 'strength_overall_away',
                'strength_attack_home', 'strength_attack_away',
                'strength_defence_home', 'strength_defence_away']

# Merge home team data
merged_teams = pd.merge(all_players_data,
                        teams_df[team_columns],
                        left_on='Team_player',
                        right_on='short_name',
                        how='left')

# Merge opponent team data (same columns but with suffix '_opponent')
merged_opponent  = pd.merge(merged_teams,
                        teams_df[team_columns],
                        left_on='vs',
                        right_on='short_name',
                        how='left',
                        suffixes=('', '_opponent'))

# Drop redundant 'short_name' columns after merge
merged_opponent .drop(columns=['short_name', 'short_name_opponent'], inplace=True)
# Apply the function to convert opponent string details
merged_opponent  = convert_opponent_string(merged_opponent )






new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]
elapsed_time3 = time.time() - start_time
st.error(f"3-Time taken by my_function: {elapsed_time3} seconds")
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict
    
    merged_dict = {}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        decoupled_list = list(set(v))
        merged_dict[k] = decoupled_list
    for i in range(1,6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict
	
sui=get_home_away_str_dict()




new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

def create_team_fdr_dataframe():
    # Create a list to store the results
    team_fdr_list = []
    for col in new_fdr_df.columns:
        # Get the values from the FDR DataFrame
        fdr_values = new_fdr_df[col].values
        # Get the corresponding teams from the fixture DataFrame
        teams = new_fixt_df[col].values
        # Combine teams with their FDR values into the list
        for team, fdr in zip(teams, fdr_values):
            # Ensure that we don't include empty FDR values or teams
            if pd.notna(fdr) and fdr > 0:  # Adjust condition as needed
                team_fdr_list.append({'team': team.strip(), 'fdr': fdr})
    # Create a DataFrame from the list
    team_fdr_df = pd.DataFrame(team_fdr_list)
    return team_fdr_df

# Example usage
team_fdr_df = create_team_fdr_dataframe()


team_fdr_map = dict(zip(team_fdr_df['team'], team_fdr_df['fdr']))
# Map the 'fdr' values to the 'merged_opponent' dataframe based on the 'Team_player' column
merged_opponent['Team_fdr'] = merged_opponent['Team_player'].map(team_fdr_map)
merged_opponent['opponent_fdr'] = merged_opponent['vs'].map(team_fdr_map)



elapsed_time4 = time.time() - start_time
st.error(f"4-Time taken by my_function: {elapsed_time4} seconds")


columns_to_convert = [
    'GW', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 
    'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 
    'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 
    'strength_overall_home', 'strength_overall_away', 'strength_attack_home', 'strength_attack_away', 
    'strength_defence_home', 'strength_defence_away', 'strength_overall_home_opponent', 
    'strength_overall_away_opponent', 'strength_attack_home_opponent', 'strength_attack_away_opponent', 
    'strength_defence_home_opponent', 'strength_defence_away_opponent', 'Team_fdr', 'opponent_fdr'
]

# Convert specified columns to numeric
for col in columns_to_convert:
    merged_opponent[col] = pd.to_numeric(merged_opponent[col], errors='coerce')  # Coerce errors to NaN

# Add season information for filtering purposes
merged_opponent['season'] = 2425

# Filter fixtures for the current gameweek
next_fixture_gw = fixtures_df[fixtures_df['event'] == ct_gw]

# Merge fixture data with team information for team_a
new_fix_gw_a = pd.merge(
    next_fixture_gw,
    teams_df[['short_name', 'name']],  # Only need 'short_name' and 'name'
    left_on='team_a',  # Match with 'team_a' column
    right_on='name', 
    how='left'
)

# Rename the 'short_name' column for clarity
new_fix_gw_a.rename(columns={'short_name': 'team_a_short_name'}, inplace=True)

# Merge fixture data with team information for team_h
new_fix_gw = pd.merge(
    new_fix_gw_a,
    teams_df[['short_name', 'name']],  # Only need 'short_name' and 'name'
    left_on='team_h',  # Match with 'team_h' column
    right_on='name', 
    how='left'
)

# Rename the 'short_name' column for clarity and drop unnecessary columns
new_fix_gw.rename(columns={'short_name': 'team_h_short_name'}, inplace=True)
new_fix_gw = new_fix_gw.drop(columns=['name_x', 'name_y'], errors='ignore')

# Append (H) for home teams and (A) for away teams
new_fix_gw['team_h_short_name'] = new_fix_gw['team_h_short_name'] + ' (H)'
new_fix_gw['team_a_short_name'] = new_fix_gw['team_a_short_name'] + ' (A)'

# Create a unique list of teams for the next gameweek
teams_next_gw = pd.concat([new_fix_gw['team_a_short_name'], new_fix_gw['team_h_short_name']]).unique()

# Prepare player data for filtering by game result
filtered_players = merged_opponent.copy()  # Create a copy to avoid modifying the original

# Split 'Result' into 'team_player_score' and 'vs_score', and convert to integers
filtered_players[['team_player_score', 'vs_score']] = filtered_players['Result'].str.split('-', expand=True)
filtered_players['team_player_score'] = filtered_players['team_player_score'].astype(int)
filtered_players['vs_score'] = filtered_players['vs_score'].astype(int)

# Drop the original 'Result' column as it's no longer needed
filtered_players.drop(columns=['Result'], axis=1, inplace=True)

# Create a new fixture dataframe with only relevant columns and rename for clarity
new_fix_gw_test = new_fix_gw[['event', 'team_h_short_name', 'team_a_short_name', 'kickoff_time']].rename(
    columns={
        'event': 'GW',
        'team_h_short_name': 'Team_home',
        'team_a_short_name': 'Team_away',
    }
)
elapsed_time5 = time.time() - start_time
st.error(f"5-Time taken by my_function: {elapsed_time5} seconds")

history_path= os.path.join(cwd, 'data', 'history', 'clean_player_2324.csv')

player_history = pd.read_csv(history_path, index_col=0)


# Concatenating the dataframes vertically
concatenated_df = pd.concat([filtered_players, player_history], ignore_index=True)

# If you want to reset the index after concatenation
concatenated_df.reset_index(drop=True, inplace=True)


new_fix_gw_test['season']=2425

elapsed_time6 = time.time() - start_time
st.error(f"6-Time taken by my_function: {elapsed_time6} seconds")


# 1. Calculate the average statistics for each team from df_player
df_player=concatenated_df
df_fixture=new_fix_gw_test

last_gw = df_player[df_player['season'] == 2425]['GW'].max()
filtered_players_fixture = df_player[
    (df_player['season'] == 2425) ]
filtered_pl = df_player[
    (df_player['season'] == 2425) & (df_player['GW'] == last_gw)
]
fit=filtered_pl[['Team_player', 'Player', 'Pos', 'Price']]

fit['team'] = fit['Team_player'].str.extract(r'([A-Za-z]+) \(')[0]

# Pre-extract team names for both 'Team_home' and 'Team_away' once
df_fixture['home_team'] = df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0]
df_fixture['away_team'] = df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0]

# Create dictionaries for home and away team mappings for 'GW', 'kickoff_time', and 'season'
home_team_mapping = df_fixture.set_index('home_team')[['GW', 'kickoff_time', 'season']].to_dict(orient='index')
away_team_mapping = df_fixture.set_index('away_team')[['GW', 'kickoff_time', 'season']].to_dict(orient='index')

# Combine home and away team mappings
combined_mapping = {}
for team, data in home_team_mapping.items():
    combined_mapping[team] = data
for team, data in away_team_mapping.items():
    if team in combined_mapping:
        # Merge the home and away data (if applicable)
        combined_mapping[team].update(data)
    else:
        combined_mapping[team] = data

# Now map the values to the 'fit' DataFrame for 'GW', 'kickoff_time', and 'season'
fit['GW'] = fit['team'].map(lambda team: combined_mapping.get(team, {}).get('GW'))
fit['kickoff_time'] = fit['team'].map(lambda team: combined_mapping.get(team, {}).get('kickoff_time'))
fit['season'] = fit['team'].map(lambda team: combined_mapping.get(team, {}).get('season'))

####################################################################

# Create dictionaries for home-to-away and away-to-home teams to find opponents
home_to_away = df_fixture.set_index('home_team')['Team_away'].to_dict()
away_to_home = df_fixture.set_index('away_team')['Team_home'].to_dict()

# Combine home and away opponent mappings into a single dictionary
combined_opponent_mapping = {}
combined_opponent_mapping.update(home_to_away)
combined_opponent_mapping.update(away_to_home)

# Map the 'vs' column in the 'fit' DataFrame to their corresponding opponent
fit['vs'] = fit['team'].map(combined_opponent_mapping)


pulga=filtered_players_fixture
columns_to_normalize = [
    'Mins','Pts', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 
    'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 'I', 
    'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out'
]

total_stats = pulga.groupby('Player')[columns_to_normalize].mean().reset_index()

df_pred = pd.merge(fit, total_stats,
                           left_on='Player', right_on='Player', how='left')


df_pred['vs_temp'] = df_pred['vs'].str.replace(r'\s?\(.*\)', '', regex=True)
df_pred['Team_player_temp'] = df_pred['Team_player'].str.replace(r'\s?\(.*\)', '', regex=True)

# Create mappings for each strength based on the team names
strength_overall_home_map = teams_df.set_index('short_name')['strength_overall_home'].to_dict()
strength_overall_away_map = teams_df.set_index('short_name')['strength_overall_away'].to_dict()
strength_attack_home_map = teams_df.set_index('short_name')['strength_attack_home'].to_dict()
strength_attack_away_map = teams_df.set_index('short_name')['strength_attack_away'].to_dict()
strength_defence_home_map = teams_df.set_index('short_name')['strength_defence_home'].to_dict()
strength_defence_away_map = teams_df.set_index('short_name')['strength_defence_away'].to_dict()

# Map the strengths for the team in `Team_player_temp`
df_pred['strength_overall_home'] = df_pred['Team_player_temp'].map(strength_overall_home_map)
df_pred['strength_overall_away'] = df_pred['Team_player_temp'].map(strength_overall_away_map)
df_pred['strength_attack_home'] = df_pred['Team_player_temp'].map(strength_attack_home_map)
df_pred['strength_attack_away'] = df_pred['Team_player_temp'].map(strength_attack_away_map)
df_pred['strength_defence_home'] = df_pred['Team_player_temp'].map(strength_defence_home_map)
df_pred['strength_defence_away'] = df_pred['Team_player_temp'].map(strength_defence_away_map)

# Map the strengths for the opponent team in `vs_temp`
df_pred['strength_overall_home_opponent'] = df_pred['vs_temp'].map(strength_overall_home_map)
df_pred['strength_overall_away_opponent'] = df_pred['vs_temp'].map(strength_overall_away_map)
df_pred['strength_attack_home_opponent'] = df_pred['vs_temp'].map(strength_attack_home_map)
df_pred['strength_attack_away_opponent'] = df_pred['vs_temp'].map(strength_attack_away_map)
df_pred['strength_defence_home_opponent'] = df_pred['vs_temp'].map(strength_defence_home_map)
df_pred['strength_defence_away_opponent'] = df_pred['vs_temp'].map(strength_defence_away_map)

# Optionally drop the 'short_name' columns for opponents if you don't need them
df_next_fixt = df_pred.drop(columns=['vs_temp','Team_player_temp'])

df_next_fixt['Team_fdr'] = df_next_fixt['Team_player'].map(team_fdr_map)
df_next_fixt['opponent_fdr'] = df_next_fixt['vs'].map(team_fdr_map)
df_next_fixt['was_home'] = df_next_fixt['Team_player'].apply(lambda x: True if '(H)' in x else False)

df_next_fixt_gw=df_next_fixt
features = ['GW', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 
            'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 'was_home', 'strength_overall_home', 'strength_overall_away', 
            'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away', 'strength_overall_home_opponent', 
            'strength_overall_away_opponent', 'strength_attack_home_opponent', 'strength_attack_away_opponent', 'strength_defence_home_opponent', 
            'strength_defence_away_opponent', 'Team_fdr', 'opponent_fdr', 'season', 'position_weight', 'home_away_weight', 'time_weight', 
            'strength_weight', 'final_weight', 'transfer_weight', 'opponent_difficulty_weight', 'penalty_risk_weight']


ssuiio=df_next_fixt_gw


ssuiio['kickoff_time'] = pd.to_datetime(ssuiio['kickoff_time'])

# 1. Position Weights
position_weights = {
    'GKP': 0.9,
    'DEF': 1.1,
    'MID': 1.3,
    'FWD': 1.5
}

# Apply weight based on position
ssuiio['position_weight'] = ssuiio['Pos'].map(position_weights)

# 2. Home/Away Game Weights
home_weight = 1.2  # Home game weight
away_weight = 1.0  # Away game weight

# Apply home/away weight
ssuiio['home_away_weight'] = ssuiio['was_home'].map({True: home_weight, False: away_weight})

# 3. Kickoff Time Weights
def assign_time_weight(kickoff_time):
    if 6 <= kickoff_time.hour < 12:
        return 0.95  # Morning games tend to have lower energy
    elif 12 <= kickoff_time.hour < 18:
        return 1.0   # Standard midday games
    elif 18 <= kickoff_time.hour < 24:
        return 1.1   # Evening games often see more action
    else:
        return 1.05  # Late-night games may see more relaxed performances

ssuiio['time_weight'] = ssuiio['kickoff_time'].apply(assign_time_weight)

# 4. Team and Opponent Strength Weights
ssuiio['team_strength_weight'] = (ssuiio['strength_overall_home'] + ssuiio['strength_attack_home'] - ssuiio['strength_defence_home']) * 1.1
ssuiio['opponent_strength_weight'] = (ssuiio['strength_overall_away_opponent'] + ssuiio['strength_attack_away_opponent'] - ssuiio['strength_defence_away_opponent'])

# Strength ratio, adjust for home/away dynamics
ssuiio['strength_weight'] = ssuiio['team_strength_weight'] / ssuiio['opponent_strength_weight']

# 5. Transfer Activity Weights
ssuiio['transfer_weight'] = ssuiio['Tran_In'] / (ssuiio['Tran_In'] + ssuiio['Tran_Out'] + 1)

# 6. Disciplinary Risk Weights
ssuiio['penalty_risk_weight'] = 1 - (ssuiio['Pen_Miss'] * 0.3 + ssuiio['YC'] * 0.1 + ssuiio['RC'] * 0.2)

# 7. Fixture Difficulty Rating (FDR) Weights
ssuiio['opponent_difficulty_weight'] = 1 / (ssuiio['opponent_fdr'] + 1)

# 8. Form Weight

# 9. Minutes Played Weight
ssuiio['minutes_weight'] = ssuiio['Mins'] / 90  # Normalize to a full game

# 10. Expected Goals (xG) and Expected Assists (xA) Weights
ssuiio['xg_weight'] = ssuiio['xG'] * 1.2  # Weight xG higher as it's a strong predictor of goals
ssuiio['xa_weight'] = ssuiio['xA'] * 1.1  # xA is also valuable for midfielders and forwards

# 11. Final Weight Calculation
ssuiio['final_weight'] = (
    ssuiio['position_weight'] * 
    ssuiio['home_away_weight'] * 
    ssuiio['time_weight'] * 
    ssuiio['strength_weight'] * 
    ssuiio['transfer_weight'] * 
    ssuiio['penalty_risk_weight'] * 
    ssuiio['opponent_difficulty_weight'] *
    ssuiio['minutes_weight'] * 
    ssuiio['xg_weight'] * 
    ssuiio['xa_weight']
)

elapsed_time7 = time.time() - start_time
st.error(f"7-Time taken by my_function: {elapsed_time7} seconds")
XX = ssuiio[features]
model_path="./Vis/pages/Prediction/xgb_model.joblib"
best_model = joblib.load(model_path)
azdazdazd=best_model.predict(XX)
ssuiio['prediction']=azdazdazd


st.write(ssuiio)

