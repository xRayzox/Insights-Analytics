import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import matplotlib.pyplot as plt  
import os 
from concurrent.futures import ThreadPoolExecutor ,ProcessPoolExecutor,as_completed
import joblib
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
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
# Retrieve and prepare player data
ele_types_data = get_bootstrap_data()['element_types']
ele_types_X_weighted = pd.DataFrame(ele_types_data)
ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)
ele_df['element_type'] = ele_df['element_type'].map(ele_types_X_weighted.set_index('id')['singular_name_short'])
ele_copy = ele_df.copy()

# Retrieve and prepare team data
teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)

# Map team IDs to names for fixture processing
team_name_mapping = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()
ele_copy['team_name'] = ele_copy['team'].map(teams_df.set_index('id')['short_name'])
ele_copy['full_name'] = ele_copy['first_name'].str.cat(ele_copy['second_name'].str.cat(ele_copy['team_name'].apply(lambda x: f" ({x})"), sep=''), sep=' ')

# Retrieve player dictionary and current season/gameweek
full_player_dict = get_player_id_dict('total_points', web_name=False)
crnt_season = get_current_season()
ct_gw = get_current_gw()

# Retrieve and process fixture data
fixture_data = get_fixture_data()
fixtures_df = pd.DataFrame(fixture_data)
fixtures_df.drop(columns='stats', inplace=True)

fixtures_df['team_h'] = fixtures_df['team_h'].replace(team_name_mapping)
fixtures_df['team_a'] = fixtures_df['team_a'].replace(team_name_mapping)
fixtures_df = fixtures_df.drop(columns=['pulse_id'])

# Format fixture dates
timezone = 'Europe/London'
fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True)
fixtures_df['local_time'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%A %d %B %Y %H:%M')
fixtures_df['local_date'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%d %A %B %Y')
fixtures_df['local_hour'] = fixtures_df['datetime'].dt.tz_convert(timezone).dt.strftime('%H:%M')

# Retrieve fixture difficulty rating data
team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
full_player_dict = get_player_id_dict('total_points', web_name=False)

def convert_score_to_result(df):
    df.loc[df['was_home'] == True, 'result'] = df['team_h_score'] \
        .astype('Int64').astype(str) \
        + '-' + df['team_a_score'].astype('Int64').astype(str)
    df.loc[df['was_home'] == False, 'result'] = df['team_a_score'] \
        .astype('Int64').astype(str) \
        + '-' + df['team_h_score'].astype('Int64').astype(str)
        
def convert_opponent_string(df):
    df.loc[df['was_home'] == True, 'vs'] = df['vs'] + ' (A)'
    df.loc[df['was_home'] == False, 'vs'] = df['vs'] + ' (H)'
    df.loc[df['was_home'] == True, 'Team_player'] = df['Team_player'] + ' (H)'
    df.loc[df['was_home'] == False, 'Team_player'] = df['Team_player'] + ' (A)'
    return df

def collate_hist_df_from_name(player_name):
    p_id = [k for k, v in full_player_dict.items() if v == player_name]
    position = ele_copy.loc[ele_copy['full_name'] == player_name, 'element_type'].iloc[0]
    Team = ele_copy.loc[ele_copy['full_name'] == player_name, 'team_name'].iloc[0]
    p_data = get_player_data(str(p_id[0]))
    p_df = pd.DataFrame(p_data['history'])
    convert_score_to_result(p_df)
    p_df.loc[p_df['result'] == '<NA>-<NA>', 'result'] = '-'
    rn_dict = {'round': 'GW','kickoff_time':'kickoff_time', 'opponent_team': 'vs', 'total_points': 'Pts',
               'minutes': 'Mins', 'goals_scored': 'GS', 'assists': 'A',
               'clean_sheets': 'CS', 'goals_conceded': 'GC', 'own_goals': 'OG',
               'penalties_saved': 'Pen_Save', 'penalties_missed': 'Pen_Miss',
               'yellow_cards': 'YC', 'red_cards': 'RC', 'saves': 'S',
               'bonus': 'B', 'bps': 'BPS', 'influence': 'I', 'creativity': 'C',
               'threat': 'T', 'ict_index': 'ICT', 'value': 'Price',
               'selected': 'SB', 'transfers_in': 'Tran_In',
               'transfers_out': 'Tran_Out', 'expected_goals': 'xG',
               'expected_assists': 'xA', 'expected_goal_involvements': 'xGI',
               'expected_goals_conceded': 'xGC', 'result': 'Result'}
    p_df.rename(columns=rn_dict, inplace=True)
    col_order = ['GW','kickoff_time', 'vs', 'Result', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA',
                 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S',
                 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB',
                 'Tran_In', 'Tran_Out', 'was_home']
    p_df = p_df[col_order]
    # map opponent teams
    
    p_df['Price'] = p_df['Price']/10
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    p_df['Pos'] = position
    p_df['Team_player'] = Team
    #convert_opponent_string(p_df)
    #p_df.drop('was_home', axis=1, inplace=True)
    #p_df.set_index('GW', inplace=True)
    p_df.sort_values('GW', ascending=False, inplace=True)
    return p_df


def collate_all_players_parallel(full_player_dict, max_workers=None):
    # Determine optimal max_workers if not provided
    if max_workers is None:
        max_workers = os.cpu_count() * 2  # Suitable for I/O-bound tasks like web scraping

    # Define a helper function to retrieve data for a single player
    def get_player_data(player_name):
        try:  # Add exception handling inside the worker function
            player_df = collate_hist_df_from_name(player_name)
            player_df['Player'] = player_name  # Add player name column
            return player_df
        except Exception as e:
            print(f"Error processing {player_name}: {e}")
            return pd.DataFrame() # Return empty DataFrame on error


    # Use ThreadPoolExecutor with a with statement for proper resource management
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and store futures in a dictionary for easier error handling
        futures = {executor.submit(get_player_data, player_name): player_name 
                   for player_name in full_player_dict.values()}

        results = []
        for future in as_completed(futures):
            player_name = futures[future]
            try:
                result_df = future.result()  # Get the result or raise an exception
                results.append(result_df)
            except Exception as e:
                print(f"Error retrieving result for {player_name}: {e}")

    # Concatenate all successful results into a single DataFrame outside the loop
    all_players_df = pd.concat(results, axis=0, ignore_index=True)  # ignore_index for cleaner index
    return all_players_df

all_players_data = collate_all_players_parallel(full_player_dict)


merged_home = pd.merge(all_players_data, teams_df[['short_name',
                                                    'strength_overall_home', 
                                                    'strength_overall_away', 
                                                    'strength_attack_home', 
                                                    'strength_attack_away', 
                                                    'strength_defence_home', 
                                                    'strength_defence_away']],
                       left_on='Team_player', 
                       right_on='short_name', 
                       how='left')

# Merge for the opponent team
merged_opponent = pd.merge(merged_home, 
                            teams_df[['short_name',
                                       'strength_overall_home', 
                                       'strength_overall_away', 
                                       'strength_attack_home', 
                                       'strength_attack_away', 
                                       'strength_defence_home', 
                                       'strength_defence_away']],
                            left_on='vs', 
                            right_on='short_name', 
                            how='left', 
                            suffixes=('', '_opponent'))

# Optionally drop the 'short_name' columns for opponents if you don't need them
merged_opponent = merged_opponent.drop(columns=['short_name', 'short_name_opponent'])
merged_opponent=convert_opponent_string(merged_opponent)


team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()

ct_gw = get_current_gw()

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]

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

team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()

ct_gw = get_current_gw()

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



columns_to_convert = ['GW', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 
                      'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 
                      'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 
                      'strength_overall_home', 'strength_overall_away', 'strength_attack_home', 'strength_attack_away', 
                      'strength_defence_home', 'strength_defence_away', 'strength_overall_home_opponent', 
                      'strength_overall_away_opponent', 'strength_attack_home_opponent', 'strength_attack_away_opponent', 
                      'strength_defence_home_opponent', 'strength_defence_away_opponent', 'Team_fdr', 'opponent_fdr']


# Convert specified columns to float
for col in columns_to_convert:
    merged_opponent[col] = pd.to_numeric(merged_opponent[col], errors='coerce')  # Convert to float and set errors to NaN if conversion fails

merged_opponent['season']=2425
next_fixture_gw = fixtures_df[fixtures_df['event']==ct_gw]


# Merge for team_a
new_fix_gw_a = pd.merge(
    next_fixture_gw,
    teams_df[['short_name', 'name']],  # Include 'name' for matching
    left_on='team_a',  # Match with team_a
    right_on='name', 
    how='left'
)

# Rename the short_name column for clarity
new_fix_gw_a.rename(columns={'short_name': 'team_a_short_name'}, inplace=True)

# Merge for team_h
new_fix_gw = pd.merge(
    new_fix_gw_a,
    teams_df[['short_name', 'name']],  # Include 'name' for matching
    left_on='team_h',  # Match with team_h
    right_on='name', 
    how='left'
)

# Rename the short_name column for clarity
new_fix_gw.rename(columns={'short_name': 'team_h_short_name'}, inplace=True)
new_fix_gw = new_fix_gw.drop(columns=['name_x', 'name_y'], errors='ignore')
new_fix_gw['team_h_short_name'] = new_fix_gw['team_h_short_name'] + ' (H)'
new_fix_gw['team_a_short_name'] = new_fix_gw['team_a_short_name'] + ' (A)'



teams_next_gw = pd.concat([new_fix_gw['team_a_short_name'], new_fix_gw['team_h_short_name']]).unique()
filtered_players = merged_opponent

filtered_players[['team_player_score', 'vs_score']] = filtered_players['Result'].str.split('-', expand=True)

# Convert the scores to integers (optional, depending on how you want to use them)
filtered_players['team_player_score'] = filtered_players['team_player_score'].astype(int)
filtered_players['vs_score'] = filtered_players['vs_score'].astype(int)
filtered_players.drop(columns=['Result'], axis=1, inplace=True)




new_fix_gw_test = new_fix_gw[['event', 'team_h_short_name', 'team_a_short_name','kickoff_time']].rename(
    columns={
        'event': 'GW',
        'team_h_short_name': 'Team_home',
        'team_a_short_name': 'Team_away',
    }
)

print('sssssssssuii')
history_path= os.path.join(cwd, 'data', 'history', 'clean_player_2324.csv')

player_history = pd.read_csv(history_path, index_col=0)


# Concatenating the dataframes vertically
concatenated_df = pd.concat([filtered_players, player_history], ignore_index=True)

# If you want to reset the index after concatenation
concatenated_df.reset_index(drop=True, inplace=True)


new_fix_gw_test['season']=2425


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

# Now, assign 'GW', 'kickoff_time', and 'season' based on matching either Team_home or Team_away in df_fixture
fit['GW'] = fit['team'].apply(
    lambda team: df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'GW'
    ].values[0] if not df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'GW'
    ].empty else None
)

fit['kickoff_time'] = fit['team'].apply(
    lambda team: df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'kickoff_time'
    ].values[0] if not df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'kickoff_time'
    ].empty else None
)

fit['season'] = fit['team'].apply(
    lambda team: df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'season'
    ].values[0] if not df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team) | 
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'season'
    ].empty else None
)
fit['vs'] = fit['team'].apply(
    lambda team: df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team), 'Team_away'
    ].values[0] if not df_fixture.loc[
        (df_fixture['Team_home'].str.extract(r'([A-Za-z]+)')[0] == team), 'Team_away'
    ].empty else df_fixture.loc[
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'Team_home'
    ].values[0] if not df_fixture.loc[
        (df_fixture['Team_away'].str.extract(r'([A-Za-z]+)')[0] == team), 'Team_home'
    ].empty else None
)



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


# List of columns to convert to float
columns_to_convert = [
    'GW', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC',
    'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C',
    'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 'strength_overall_home',
    'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
    'strength_defence_home', 'strength_defence_away',
    'strength_overall_home_opponent', 'strength_overall_away_opponent',
    'strength_attack_home_opponent', 'strength_attack_away_opponent',
    'strength_defence_home_opponent', 'strength_defence_away_opponent',
    'Team_fdr', 'opponent_fdr'
]

df_next_fixt_gw[columns_to_convert] = df_next_fixt_gw[columns_to_convert].astype(float)
df_player[columns_to_convert] = df_player[columns_to_convert].astype(float)
# Make a copy of X to apply weights
X_weighted = df_player.copy()


print('sssssssssuii2')
# Define the objective function for Optuna
def objective_optuna(trial):
    # Define the hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.15, 0.25),
        'max_depth': trial.suggest_int('max_depth', 2,8),
        'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 3),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.01, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1),
    }

    model = XGBRegressor(objective='reg:squarederror', random_state=42,**params)

    # Use K-fold cross-validation to better estimate performance
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mean_mse = -mse_scores.mean()  # Take the negative because cross_val_score returns negative MSE

    return mean_mse

# Function to perform hyperparameter tuning using Optuna
def auto_tune_hyperparameters(X, y, n_trials=5):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if previous Optuna study exists
    study_path = "./Vis/pages/Prediction/optuna_study.joblib"
    if os.path.exists(study_path):
        study = joblib.load(study_path)
        print("Loaded existing Optuna study for incremental tuning.")
    else:
        study = optuna.create_study(direction='minimize')
        print("Starting a new Optuna study for hyperparameter tuning.")

    # Run optimization
    study.optimize(objective_optuna, n_trials=n_trials)

    # Save the study for future reference or incremental tuning
    joblib.dump(study, study_path)
    print("Optuna study saved.")

    # Retrieve the best parameters
    best_params = study.best_params
    return best_params

# Function to train, evaluate, and save the model if it improves performance
def train_and_save_model(X, y, model_path="./Vis/pages/Prediction/xgb_model.joblib", score_path="./Vis/pages/Prediction/best_score.joblib"):
    # Check if a model and score already exist
    if os.path.exists(model_path) and os.path.exists(score_path):
        model = joblib.load(model_path)
        best_score = joblib.load(score_path)
        print(f"Loaded existing model with best score: {best_score:.4f}")
    else:
        best_score = float("inf")  # Initialize to infinity for first-time training

    # Tune hyperparameters if no previous params are found
    best_params = auto_tune_hyperparameters(X, y)
    new_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    print("Starting new model training with tuned hyperparameters.")

    # Train the model with early stopping
    new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Evaluate the new model on the test set
    y_pred = new_model.predict(X_test)
    new_score = mean_squared_error(y_test, y_pred)

    # Compare and save the best model based on the score
    if new_score < best_score:
        print(f"New model improves from {best_score:.4f} to {new_score:.4f}. Saving new model.")
        joblib.dump(new_model, model_path)
        joblib.dump(new_score, score_path)
        print("New model and score saved.")
        return new_model
    else:
        print(f"New model score {new_score:.4f} did not improve. Retaining the existing model.")
        return model


X_weighted['kickoff_time'] = pd.to_datetime(X_weighted['kickoff_time'])

# 1. Position Weights
position_weights = {
    'GKP': 0.9,
    'DEF': 1.1,
    'MID': 1.3,
    'FWD': 1.5
}

# Apply weight based on position
X_weighted['position_weight'] = X_weighted['Pos'].map(position_weights)

# 2. Home/Away Game Weights
home_weight = 1.2  # Home game weight
away_weight = 1.0  # Away game weight

# Apply home/away weight
X_weighted['home_away_weight'] = X_weighted['was_home'].map({True: home_weight, False: away_weight})

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

X_weighted['time_weight'] = X_weighted['kickoff_time'].apply(assign_time_weight)

# 4. Team and Opponent Strength Weights
X_weighted['team_strength_weight'] = (X_weighted['strength_overall_home'] + X_weighted['strength_attack_home'] - X_weighted['strength_defence_home']) * 1.1
X_weighted['opponent_strength_weight'] = (X_weighted['strength_overall_away_opponent'] + X_weighted['strength_attack_away_opponent'] - X_weighted['strength_defence_away_opponent'])

# Strength ratio, adjust for home/away dynamics
X_weighted['strength_weight'] = X_weighted['team_strength_weight'] / X_weighted['opponent_strength_weight']

# 5. Transfer Activity Weights
X_weighted['transfer_weight'] = X_weighted['Tran_In'] / (X_weighted['Tran_In'] + X_weighted['Tran_Out'] + 1)

# 6. Disciplinary Risk Weights
X_weighted['penalty_risk_weight'] = 1 - (X_weighted['Pen_Miss'] * 0.3 + X_weighted['YC'] * 0.1 + X_weighted['RC'] * 0.2)

# 7. Fixture Difficulty Rating (FDR) Weights
X_weighted['opponent_difficulty_weight'] = 1 / (X_weighted['opponent_fdr'] + 1)


# 9. Minutes Played Weight
X_weighted['minutes_weight'] = X_weighted['Mins'] / 90  # Normalize to a full game

# 10. Expected Goals (xG) and Expected Assists (xA) Weights
X_weighted['xg_weight'] = X_weighted['xG'] * 1.2  # Weight xG higher as it's a strong predictor of goals
X_weighted['xa_weight'] = X_weighted['xA'] * 1.1  # xA is also valuable for midfielders and forwards

# 11. Final Weight Calculation
X_weighted['final_weight'] = (
    X_weighted['position_weight'] * 
    X_weighted['home_away_weight'] * 
    X_weighted['time_weight'] * 
    X_weighted['strength_weight'] * 
    X_weighted['transfer_weight'] * 
    X_weighted['penalty_risk_weight'] * 
    X_weighted['opponent_difficulty_weight'] *
    X_weighted['minutes_weight'] * 
    X_weighted['xg_weight'] * 
    X_weighted['xa_weight']
)


features = ['GW', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 
            'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 'was_home', 'strength_overall_home', 'strength_overall_away', 
            'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away', 'strength_overall_home_opponent', 
            'strength_overall_away_opponent', 'strength_attack_home_opponent', 'strength_attack_away_opponent', 'strength_defence_home_opponent', 
            'strength_defence_away_opponent', 'Team_fdr', 'opponent_fdr', 'season', 'position_weight', 'home_away_weight', 'time_weight', 
            'strength_weight', 'final_weight', 'transfer_weight', 'opponent_difficulty_weight', 'penalty_risk_weight']
X = X_weighted[features]
y = X_weighted['Pts']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = train_and_save_model(X_train, y_train)



