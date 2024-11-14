import streamlit as st
import warnings
import pandas as pd
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np

warnings.filterwarnings("ignore")

# Define paths and add FPL directory to system path
cwd = os.getcwd()
fpl_path = os.path.join(cwd, 'FPL')
sys.path.append(fpl_path)

from fpl_api_collection import (get_bootstrap_data, get_current_gw,
                                get_fixt_dfs, get_fixture_data, get_player_id_dict,
                                get_current_season, get_player_data)

# --- Data Retrieval and Preprocessing ---

# Player data
ele_types_data = get_bootstrap_data()['element_types']
ele_data = get_bootstrap_data()['elements']
ele_df = pd.DataFrame(ele_data)
ele_df['element_type'] = ele_df['element_type'].map(pd.DataFrame(ele_types_data).set_index('id')['singular_name_short'])

# Team data
teams_data = get_bootstrap_data()['teams']
teams_df = pd.DataFrame(teams_data)
team_name_mapping = teams_df.set_index('id')['name'].to_dict()
ele_df['team_name'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])
ele_df['full_name'] = ele_df['first_name'].str.cat(ele_df['second_name'], sep=' ').str.cat(ele_df['team_name'].apply(lambda x: f" ({x})"), sep=' ')

# Gameweek and Season info
full_player_dict = get_player_id_dict('total_points', web_name=False)
crnt_season = get_current_season()
ct_gw = get_current_gw()


# Fixture data
fixtures_df = pd.DataFrame(get_fixture_data()).drop(columns='stats')
fixtures_df['team_h'] = fixtures_df['team_h'].replace(team_name_mapping)
fixtures_df['team_a'] = fixtures_df['team_a'].replace(team_name_mapping)
fixtures_df = fixtures_df.drop(columns=['pulse_id'])


# Fixture Date Formatting
timezone = 'Europe/London'
fixtures_df['datetime'] = pd.to_datetime(fixtures_df['kickoff_time'], utc=True).dt.tz_convert(timezone)
fixtures_df['local_time'] = fixtures_df['datetime'].dt.strftime('%A %d %B %Y %H:%M')
fixtures_df['local_date'] = fixtures_df['datetime'].dt.strftime('%d %A %B %Y')
fixtures_df['local_hour'] = fixtures_df['datetime'].dt.strftime('%H:%M')



# --- Helper Functions ---

def convert_score_to_result(df):
    df['result'] = np.where(df['was_home'], 
                           df['team_h_score'].astype('Int64').astype(str) + '-' + df['team_a_score'].astype('Int64').astype(str),
                           df['team_a_score'].astype('Int64').astype(str) + '-' + df['team_h_score'].astype('Int64').astype(str))
    df['result'] = df['result'].replace('<NA>-<NA>', '-')

def convert_opponent_string(df):
    df['vs'] = np.where(df['was_home'], df['vs'] + ' (A)', df['vs'] + ' (H)')
    df['Team_player'] = np.where(df['was_home'], df['Team_player'] + ' (H)', df['Team_player'] + ' (A)')
    return df

def collate_hist_df_from_name(player_name):
    p_id = next((k for k, v in full_player_dict.items() if v == player_name), None)
    if not p_id:
        return pd.DataFrame()  # Return empty DataFrame if player not found

    try:
        position = ele_df.loc[ele_df['full_name'] == player_name, 'element_type'].values[0]
        team = ele_df.loc[ele_df['full_name'] == player_name, 'team_name'].values[0]
    except IndexError:  # Handle the case where no match is found
        print(f"Player {player_name} not found in ele_df.")  # Or log the error
        return pd.DataFrame()
    p_data = get_player_data(str(p_id))

    if 'history' not in p_data:
        return pd.DataFrame()

    p_df = pd.DataFrame(p_data['history'])
    convert_score_to_result(p_df)

    rn_dict = {'round': 'GW', 'kickoff_time': 'kickoff_time', 'opponent_team': 'vs', 'total_points': 'Pts',
               'minutes': 'Mins', 'goals_scored': 'GS', 'assists': 'A', 'clean_sheets': 'CS',
               'goals_conceded': 'GC', 'own_goals': 'OG', 'penalties_saved': 'Pen_Save',
               'penalties_missed': 'Pen_Miss', 'yellow_cards': 'YC', 'red_cards': 'RC', 'saves': 'S',
               'bonus': 'B', 'bps': 'BPS', 'influence': 'I', 'creativity': 'C', 'threat': 'T',
               'ict_index': 'ICT', 'value': 'Price', 'selected': 'SB', 'transfers_in': 'Tran_In',
               'transfers_out': 'Tran_Out', 'expected_goals': 'xG', 'expected_assists': 'xA',
               'expected_goal_involvements': 'xGI', 'expected_goals_conceded': 'xGC', 'result': 'Result'}
    p_df = p_df.rename(columns=rn_dict)[['GW','kickoff_time', 'vs', 'Result', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA',
                   'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS',
                   'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 'was_home']]

    p_df['Price'] = p_df['Price'] / 10
    p_df['vs'] = p_df['vs'].map(teams_df.set_index('id')['short_name'])
    p_df['Pos'] = position
    p_df['Team_player'] = team
    p_df['Player'] = player_name
    p_df.sort_values('GW', ascending=False, inplace=True)
    return p_df


def collate_all_players_parallel(player_dict):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(collate_hist_df_from_name, player_name): player_name for player_name in player_dict.values()}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    return pd.concat(results, ignore_index=True)



def get_home_away_str_dict():
    team_fdr_df, team_fixt_df, _, _ = get_fixt_dfs()
    new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
    new_fixt_cols = [f'GW{col}' for col in new_fixt_df.columns]
    new_fixt_df.columns = new_fixt_cols
    new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)].rename(columns=dict(zip(team_fdr_df.columns,new_fixt_cols)) )


    result_dict = {}
    for col in new_fdr_df.columns:
        values = new_fdr_df[col].values
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df[col] = np.where(new_fixt_df[col].str.len() <= 7,  new_fixt_df[col].str.pad(width=max_length+9, side='both', fillchar=' '), new_fixt_df[col])
        strings = new_fixt_df[col].values
        value_dict = {}
        for value, string in zip(values, strings):

            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict
    
    merged_dict = {}
    for dict1 in result_dict.values():
        for key, value in dict1.items():
            merged_dict.setdefault(key, []).extend(value)

    merged_dict = {k: list(set(v)) for k, v in merged_dict.items()}

    for i in range(1,6):
        merged_dict.setdefault(i, [])
    return merged_dict
	

def create_team_fdr_dataframe():
    team_fdr_list = []
    team_fdr_df, team_fixt_df, _, _ = get_fixt_dfs()
    new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
    new_fixt_cols = [f'GW{col}' for col in new_fixt_df.columns]
    new_fixt_df.columns = new_fixt_cols
    new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)].rename(columns=dict(zip(team_fdr_df.columns,new_fixt_cols)))



    for col in new_fdr_df.columns:
        fdr_values = new_fdr_df[col].values
        teams = new_fixt_df[col].values
        for team, fdr in zip(teams, fdr_values):
            if pd.notna(fdr) and fdr > 0:
                team_fdr_list.append({'team': team.strip(), 'fdr': fdr})

    return pd.DataFrame(team_fdr_list)


def prepare_data_for_prediction(player_history_path):
    all_players_data = collate_all_players_parallel(full_player_dict)

    # Merge team strength data
    merged_data = pd.merge(all_players_data, teams_df[['short_name', 'strength_overall_home', 'strength_overall_away',
                                                     'strength_attack_home', 'strength_attack_away',
                                                     'strength_defence_home', 'strength_defence_away']],
                          left_on='Team_player', right_on='short_name', how='left').drop(columns=['short_name'])

    merged_data = pd.merge(merged_data, teams_df[['short_name', 'strength_overall_home', 'strength_overall_away',
                                                'strength_attack_home', 'strength_attack_away',
                                                'strength_defence_home', 'strength_defence_away']],
                          left_on='vs', right_on='short_name', how='left', suffixes=('', '_opponent')).drop(columns=['short_name'])


    merged_data = convert_opponent_string(merged_data)

    team_fdr_df = create_team_fdr_dataframe()
    team_fdr_map = team_fdr_df.set_index('team')['fdr'].to_dict()
    merged_data['Team_fdr'] = merged_data['Team_player'].map(team_fdr_map)
    merged_data['opponent_fdr'] = merged_data['vs'].map(team_fdr_map)


    # Convert columns to numeric
    numeric_cols = ['GW', 'Pts', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG',
                   'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In',
                   'Tran_Out', 'strength_overall_home', 'strength_overall_away', 'strength_attack_home',
                   'strength_attack_away', 'strength_defence_home', 'strength_defence_away',
                   'strength_overall_home_opponent', 'strength_overall_away_opponent', 'strength_attack_home_opponent',
                   'strength_attack_away_opponent', 'strength_defence_home_opponent', 'strength_defence_away_opponent',
                   'Team_fdr', 'opponent_fdr']
    merged_data[numeric_cols] = merged_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    merged_data['season'] = 2425

    # Merge with next fixture data
    next_fixture_gw = fixtures_df[fixtures_df['event'] == ct_gw]
    next_fixture_gw = pd.merge(next_fixture_gw, teams_df[['short_name', 'name']], left_on='team_a', right_on='name', how='left').rename(columns={'short_name': 'team_a_short_name'}).drop(columns=['name'])
    next_fixture_gw = pd.merge(next_fixture_gw, teams_df[['short_name', 'name']], left_on='team_h', right_on='name', how='left').rename(columns={'short_name': 'team_h_short_name'}).drop(columns=['name'])
    next_fixture_gw['team_h_short_name'] = next_fixture_gw['team_h_short_name'] + ' (H)'
    next_fixture_gw['team_a_short_name'] = next_fixture_gw['team_a_short_name'] + ' (A)'

    teams_next_gw = pd.concat([next_fixture_gw['team_a_short_name'], next_fixture_gw['team_h_short_name']]).unique()
    filtered_players = merged_data



    filtered_players[['team_player_score', 'vs_score']] = filtered_players['Result'].str.split('-', expand=True).astype(int)
    filtered_players = filtered_players.drop(columns=['Result'])


    # Load player history and concatenate
    player_history = pd.read_csv(player_history_path, index_col=0)
    concatenated_df = pd.concat([filtered_players, player_history], ignore_index=True)


    new_fix_gw_test = next_fixture_gw[['event', 'team_h_short_name', 'team_a_short_name','kickoff_time']].rename(
        columns={
            'event': 'GW',
            'team_h_short_name': 'Team_home',
            'team_a_short_name': 'Team_away',
        }
    )

    new_fix_gw_test['season']=2425



    #Prepare data for prediction




    last_gw = concatenated_df[concatenated_df['season'] == 2425]['GW'].max()
    filtered_players_fixture = concatenated_df[(concatenated_df['season'] == 2425)]
    filtered_pl = concatenated_df[(concatenated_df['season'] == 2425) & (concatenated_df['GW'] == last_gw)]
    fit = filtered_pl[['Team_player', 'Player', 'Pos', 'Price']]
    fit['team'] = fit['Team_player'].str.extract(r'([A-Za-z]+)')[0]
    fit = pd.merge(fit, new_fix_gw_test, left_on='team', right_on=new_fix_gw_test['Team_home'].str.extract(r'([A-Za-z]+)')[0],how='left')

    fit = pd.merge(fit, new_fix_gw_test, left_on='team', right_on=new_fix_gw_test['Team_away'].str.extract(r'([A-Za-z]+)')[0],how='left',suffixes=('_home','_away'))

    def fill_missing(row):
        for col in ['GW', 'kickoff_time', 'season','Team_home','Team_away']:
            if pd.isna(row[col + '_home']):
               return row[col + '_away']
            else:
                return row[col + '_home']
    

    for col in ['GW', 'kickoff_time', 'season','Team_home','Team_away']:
         fit[col] = fit.apply(fill_missing, axis=1)

    fit['vs'] = np.where(fit['Team_player'].str.endswith('(H)'), fit['Team_away'], fit['Team_home'])

    fit = fit.drop(columns=[col for col in fit.columns if col.endswith('_home') or col.endswith('_away')])

    columns_to_normalize = ['Mins', 'Pts', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG',
                            'Pen_Save', 'S', 'YC', 'RC', 'B', 'BPS', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out']
    total_stats = filtered_players_fixture.groupby('Player')[columns_to_normalize].mean().reset_index()
    df_pred = pd.merge(fit, total_stats, on='Player', how='left')

    df_pred['vs_temp'] = df_pred['vs'].str.replace(r'\s?\(.*\)', '', regex=True)
    df_pred['Team_player_temp'] = df_pred['Team_player'].str.replace(r'\s?\(.*\)', '', regex=True)

    strength_cols = ['strength_overall_home', 'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']
    for col in strength_cols:
        df_pred[col] = df_pred['Team_player_temp'].map(teams_df.set_index('short_name')[col].to_dict())
        df_pred[col + '_opponent'] = df_pred['vs_temp'].map(teams_df.set_index('short_name')[col].to_dict())

    df_next_fixt = df_pred.drop(columns=['vs_temp', 'Team_player_temp','team'])

    df_next_fixt['Team_fdr'] = df_next_fixt['Team_player'].map(team_fdr_map)
    df_next_fixt['opponent_fdr'] = df_next_fixt['vs'].map(team_fdr_map)

    df_next_fixt['was_home'] = df_next_fixt['Team_player'].str.endswith('(H)')

    return df_next_fixt

def calculate_weights(df):
    position_weights = {'GKP': 0.9, 'DEF': 1.1, 'MID': 1.3, 'FWD': 1.5}
    df['position_weight'] = df['Pos'].map(position_weights)
    df['home_away_weight'] = np.where(df['was_home'], 1.2, 1.0)

    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df['time_weight'] = df['kickoff_time'].apply(lambda x: 0.95 if 6 <= x.hour < 12 else 1.0 if 12 <= x.hour < 18 else 1.1 if 18 <= x.hour < 24 else 1.05)
    df['team_strength_weight'] = (df['strength_overall_home'] + df['strength_attack_home'] - df['strength_defence_home']) * 1.1
    df['opponent_strength_weight'] = (df['strength_overall_away_opponent'] + df['strength_attack_away_opponent'] - df['strength_defence_away_opponent'])
    df['strength_weight'] = df['team_strength_weight'] / df['opponent_strength_weight']
    df['transfer_weight'] = df['Tran_In'] / (df['Tran_In'] + df['Tran_Out'] + 1)
    df['penalty_risk_weight'] = 1 - (df['Pen_Miss'] * 0.3 + df['YC'] * 0.1 + df['RC'] * 0.2)
    df['opponent_difficulty_weight'] = 1 / (df['opponent_fdr'] + 1)
    df['minutes_weight'] = df['Mins'] / 90
    df['xg_weight'] = df['xG'] * 1.2
    df['xa_weight'] = df['xA'] * 1.1
    df['final_weight'] = (df['position_weight'] * df['home_away_weight'] * df['time_weight'] *
                         df['strength_weight'] * df['transfer_weight'] * df['penalty_risk_weight'] *
                         df['opponent_difficulty_weight'] * df['minutes_weight'] * df['xg_weight'] * df['xa_weight'])
    return df


# --- Main Execution ---

history_path = os.path.join(cwd, 'data', 'history', 'clean_player_2324.csv')
df_next_fixt = prepare_data_for_prediction(history_path)

features = ['GW', 'Mins', 'GS', 'xG', 'A', 'xA', 'xGI', 'Pen_Miss', 'CS', 'GC', 'xGC', 'OG', 'Pen_Save', 'S', 'YC',
            'RC', 'B', 'BPS', 'Price', 'I', 'C', 'T', 'ICT', 'SB', 'Tran_In', 'Tran_Out', 'was_home',
            'strength_overall_home', 'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away', 'strength_overall_home_opponent',
            'strength_overall_away_opponent', 'strength_attack_home_opponent', 'strength_attack_away_opponent',
            'strength_defence_home_opponent', 'strength_defence_away_opponent', 'Team_fdr', 'opponent_fdr', 'season']




df_next_fixt_gw = calculate_weights(df_next_fixt)
XX = df_next_fixt_gw[features] # Make sure all features are present.



model_path = "./Vis/pages/Prediction/xgb_model.joblib" 
best_model = joblib.load(model_path)
predictions = best_model.predict(XX)
df_next_fixt_gw['prediction'] = predictions


st.write(df_next_fixt_gw)