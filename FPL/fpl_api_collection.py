import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import polars as pl

base_url = 'https://fantasy.premierleague.com/api/'

# Function to get general data (bootstrap data) from the FPL API

#@st.cache_data(persist="disk")
def get_bootstrap_data() -> dict:
    resp = requests.get(f'{base_url}bootstrap-static/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get fixture data (upcoming matches)

#@st.cache_data(persist="disk")
def get_fixture_data() -> dict:
    resp = requests.get(f'{base_url}fixtures/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()
    


#@st.cache_data(persist="disk")
def get_player_data(player_id) -> dict:
    resp = requests.get(f'{base_url}element-summary/{player_id}/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get FPL manager details


#@st.cache_data(persist="disk")
def get_manager_details(manager_id) -> dict:
    resp = requests.get(f'{base_url}entry/{manager_id}/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get FPL manager's history (past seasons' performances)

#@st.cache_data(persist="disk")
def get_manager_history_data(manager_id) -> dict:
    resp = requests.get(f'{base_url}entry/{manager_id}/history/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get a manager's selected team for a given gameweek (GW)

#@st.cache_data(persist="disk")
def get_manager_team_data(manager_id, gw):
    resp = requests.get(f'{base_url}entry/{manager_id}/event/{gw}/picks/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get the total number of FPL players (managers)
def get_total_fpl_players():
    return get_bootstrap_data()['total_players']



# Function to filter out players who have left their clubs or are on loan
def remove_moved_players(df: pl.DataFrame) -> pl.DataFrame:
    # List of strings to check in the 'news' column
    strings = ['loan', 'Loan', 'Contract cancelled', 'Left the club',
               'Permanent', 'Released', 'Signed for', 'Transferred',
               'Season long', 'Not training', 'permanent', 'transferred']
    
    # Use the 'str.contains' function from Polars to filter out rows
    pattern = '|'.join(strings)
    df_copy = df.filter(~df['news'].str.contains(pattern, case=False))
    
    return df_copy


# Function to create a dictionary mapping player IDs to their names
def get_player_id_dict(order_by_col, web_name=True) -> dict:
    # Get the player and team data
    elements = get_bootstrap_data()['elements']
    teams = get_bootstrap_data()['teams']
    
    # Convert the JSON data to Polars DataFrames
    ele_df = pl.DataFrame(elements)
    teams_df = pl.DataFrame(teams)
    
    # Remove moved players
    ele_df = remove_moved_players(ele_df)
    
    # Join teams_df with ele_df based on the team ID and get the team names
    teams_df = teams_df.select(['id', 'short_name']).rename({'id': 'team'})
    ele_df = ele_df.join(teams_df, on='team', how='left')
    
    # Sort by the specified column
    ele_df = ele_df.sort(by=order_by_col, reverse=True)
    
    # Create the id_dict based on web_name or full_name
    if web_name:
        id_dict = dict(zip(ele_df['id'], ele_df['web_name']))
    else:
        ele_df = ele_df.with_columns(
            (ele_df['first_name'] + ' ' + ele_df['second_name'] + ' (' + ele_df['short_name'] + ')').alias('full_name')
        )
        id_dict = dict(zip(ele_df['id'], ele_df['full_name']))
    
    return id_dict

# Function to gather historic gameweek data for all players

def collate_player_hist() -> pl.DataFrame:
    res = []
    p_dict = get_player_id_dict()

    # Use ThreadPoolExecutor to fetch player data in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_player_data, p_id): p_name for p_id, p_name in p_dict.items()}
        for future in futures:
            p_name = futures[future]
            try:
                resp = future.result()
                # Append history data to the result list
                res.append(resp['history'])
            except Exception as e:
                print(f'Request to {p_name} data failed: {e}')

    # Convert the list of histories into a Polars DataFrame
    # Flatten the list if necessary to make sure it is in the right structure
    return pl.DataFrame(res)


# Team, games_played, wins, losses, draws, goals_for, goals_against, GD,
# PTS, Form? [W,W,L,D,W]

# Function to create a league table based on fixture results
def get_league_table() -> pl.DataFrame:
    fixt_df = pl.DataFrame(get_fixture_data())  # Assuming this is already a JSON response
    teams_df = pl.DataFrame(get_bootstrap_data()['teams'])
    
    teams_id_list = teams_df['id'].unique().to_list()
    df_list = []

    for t_id in teams_id_list:
        # Extract home and away data
        home_data = fixt_df.filter(fixt_df['team_h'] == t_id)
        away_data = fixt_df.filter(fixt_df['team_a'] == t_id)

        # Add 'was_home' column
        home_data = home_data.with_columns(pl.lit(True).alias('was_home'))
        away_data = away_data.with_columns(pl.lit(False).alias('was_home'))

        # Combine home and away data
        df = pl.concat([home_data, away_data])

        # Sort by 'event'
        df = df.sort('event')

        # Assign match results (win, draw, loss)
        df = df.with_columns([
            (pl.when((df['was_home'] == True) & (df['team_h_score'] > df['team_a_score'])).then(True).otherwise(False)).alias('win'),
            (pl.when((df['was_home'] == False) & (df['team_a_score'] > df['team_h_score'])).then(True).otherwise(False)).alias('win'),
            (pl.when(df['team_h_score'] == df['team_a_score']).then(True).otherwise(False)).alias('draw'),
            (pl.when((df['was_home'] == True) & (df['team_h_score'] < df['team_a_score'])).then(True).otherwise(False)).alias('loss'),
            (pl.when((df['was_home'] == False) & (df['team_a_score'] < df['team_h_score'])).then(True).otherwise(False)).alias('loss'),
            (pl.when(df['was_home'] == True).then(df['team_h_score']).otherwise(df['team_a_score'])).alias('gf'),
            (pl.when(df['was_home'] == True).then(df['team_a_score']).otherwise(df['team_h_score'])).alias('ga'),
            (pl.when(df['win'] == True).then('W').otherwise(None)).alias('result'),
            (pl.when(df['draw'] == True).then('D').otherwise(None)).alias('result'),
            (pl.when(df['loss'] == True).then('L').otherwise(None)).alias('result')
        ])

        # Clean sheet calculation
        df = df.with_columns([
            (pl.when((df['was_home'] == True) & (df['team_a_score'] == 0)).then(True).otherwise(False)).alias('clean_sheet'),
            (pl.when((df['was_home'] == False) & (df['team_h_score'] == 0)).then(True).otherwise(False)).alias('clean_sheet')
        ])

        # Calculate stats for the team
        ws = df.filter(df['win'] == True).shape[0]
        ds = df.filter(df['draw'] == True).shape[0]
        finished_df = df.filter(df['finished'] == True)

        l_data = {
            'id': [t_id],
            'GP': [finished_df.shape[0]],
            'W': [ws],
            'D': [ds],
            'L': [df.filter(df['loss'] == True).shape[0]],
            'GF': [df['gf'].sum()],
            'GA': [df['ga'].sum()],
            'GD': [df['gf'].sum() - df['ga'].sum()],
            'CS': [df['clean_sheet'].sum()],
            'Pts': [(ws * 3) + ds],
            'Form': [''.join(finished_df['result'].tail(5).to_list())]
        }
        
        # Convert the dictionary to a Polars DataFrame and append it to the df_list
        df_list.append(pl.DataFrame(l_data))

    # Concatenate all team data frames
    league_df = pl.concat(df_list)

    # Map the team short names
    league_df = league_df.with_columns(
        teams_df['short_name'].to_series().to_list().alias('team')
    )

    league_df = league_df.drop('id')

    # Sort the DataFrame
    league_df = league_df.sort(by=['Pts', 'GD', 'GF', 'GA'], reverse=[True, True, True, True])

    # Set 'team' as the index
    league_df = league_df.set_index('team')

    # Calculate per game stats
    league_df = league_df.with_columns([
        (league_df['Pts'] / league_df['GP']).round(2).alias('Pts/Game'),
        (league_df['GF'] / league_df['GP']).round(2).alias('GF/Game'),
        (league_df['GA'] / league_df['GP']).round(2).alias('GA/Game'),
        (league_df['CS'] / league_df['GP']).round(2).alias('CS/Game')
    ])

    return league_df




def get_current_gw() -> int:
    events_df = pl.DataFrame(get_bootstrap_data()['events'])
    
    # Filter the dataframe to get the current gameweek where 'is_next' is True
    current_gw = events_df.filter(events_df['is_next'] == True).select('id').to_pandas().iloc[0]['id']
    
    return current_gw


def get_current_season() -> str:
    events_df = pl.DataFrame(get_bootstrap_data()['events'])
    
    # Extract year and month portions from the 'deadline_time' column
    id_first = events_df['deadline_time'].str.slice(0, 4).to_list()[0]
    id_last = events_df['deadline_time'].str.slice(2, 4).to_list()[-1]
    
    # Combine to form the current season string
    current_season = str(id_first) + '/' + str(id_last)
    return current_season


def get_fixture_dfs():
    # doubles??
    fixt_df = pl.DataFrame(get_fixture_data())
    teams_df = pl.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().to_list()
    
    # Don't need to worry about double fixtures just yet!
    fixt_df = fixt_df.with_columns([
        fixt_df['team_h'].map(teams_df.set_index('id')['short_name'].to_dict()).alias('team_h'),
        fixt_df['team_a'].map(teams_df.set_index('id')['short_name'].to_dict()).alias('team_a')
    ])
    
    gw_dict = {i: num for i in range(1, 381) for num in range(1, 39) for _ in range(10)}
    fixt_df = fixt_df.with_columns([
        fixt_df['id'].map(gw_dict).alias('event_lock')
    ])
    
    team_fdr_data = []
    team_fixt_data = []
    
    for team in teams_list:
        home_data = fixt_df.filter(fixt_df['team_h'] == team)
        away_data = fixt_df.filter(fixt_df['team_a'] == team)
        
        home_data = home_data.with_columns([
            pl.lit(True).alias('was_home')
        ])
        away_data = away_data.with_columns([
            pl.lit(False).alias('was_home')
        ])
        
        df = pl.concat([home_data, away_data])
        df = df.sort('event_lock')

        h_filt = (df['team_h'] == team) & df['event'].is_not_null()
        a_filt = (df['team_a'] == team) & df['event'].is_not_null()

        df = df.with_columns([
            pl.when(h_filt).then(df['team_a'] + ' (H)').otherwise(df['next']).alias('next'),
            pl.when(a_filt).then(df['team_h'] + ' (A)').otherwise(df['next']).alias('next'),
            pl.when(df['event'].is_null()).then('BLANK').otherwise(df['next']).alias('next')
        ])
        
        df = df.with_columns([
            pl.when(h_filt).then(df['team_h_difficulty']).otherwise(df['next_fdr']).alias('next_fdr'),
            pl.when(a_filt).then(df['team_a_difficulty']).otherwise(df['next_fdr']).alias('next_fdr')
        ])

        team_fixt_data.append(df[['next']].with_columns([pl.lit(team).alias('team')]))
        team_fdr_data.append(df[['next_fdr']].with_columns([pl.lit(team).alias('team')]))

    team_fdr_df = pl.concat(team_fdr_data).set_index(0)
    team_fixt_df = pl.concat(team_fixt_data).set_index(0)
    
    return team_fdr_df, team_fixt_df


import polars as pl

def get_fixt_dfs():
    fixt_df = pl.DataFrame(get_fixture_data())
    teams_df = pl.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().to_list()
    league_df = pl.DataFrame(get_league_table()).reset_index()
    
    # Pre-map team names to avoid repeated mapping
    team_map = teams_df.set_index('id')['short_name'].to_dict()
    fixt_df = fixt_df.with_columns([
        fixt_df['team_h'].map(team_map).alias('team_h'),
        fixt_df['team_a'].map(team_map).alias('team_a')
    ])
    
    # Pre-create the event_dict to avoid repeated generation
    gw_dict = {i: num for i in range(1, 381) for num in range(1, 39) for _ in range(10)}
    fixt_df = fixt_df.with_columns([
        fixt_df['id'].map(gw_dict).alias('event_lock')
    ])

    team_fdr_data, team_fixt_data, team_ga_data, team_gf_data = [], [], [], []

    for team in teams_list:
        # Filter home and away data once
        home_data = fixt_df.filter(fixt_df['team_h'] == team).with_columns([
            pl.lit(True).alias('was_home')
        ])
        away_data = fixt_df.filter(fixt_df['team_a'] == team).with_columns([
            pl.lit(False).alias('was_home')
        ])
        
        # Concatenate home and away data
        df = pl.concat([home_data, away_data]).sort('kickoff_time')

        # Vectorized operations for 'next' and 'next_fdr' columns
        df = df.with_columns([
            pl.when(df['team_h'] == team).then(df['team_a'] + ' (H)').otherwise(df['team_h'] + ' (A)').alias('next'),
            pl.when(df['team_h'] == team).then(df['team_h_difficulty']).otherwise(df['team_a_difficulty']).alias('next_fdr')
        ])

        # Merge the league data efficiently
        df = df.join(league_df[['team', 'GA/Game', 'GF/Game']], on='team', how='left')
        
        # Grouping and aggregation
        event_df = pl.DataFrame({'event': [num for num in range(1, 39)]})
        
        dedup_df = df.groupby('event').agg([
            pl.col('next').apply(lambda x: ' + '.join(x)).alias('next')
        ])
        
        dedup_fdr_df = df.groupby('event').agg([
            pl.col('next_fdr').mean().alias('next_fdr'),
            pl.col('GA/Game').mean().alias('GA/Game'),
            pl.col('GF/Game').mean().alias('GF/Game')
        ])
        
        # Merge aggregated results
        dedup_df = dedup_df.join(dedup_fdr_df, on='event', how='left')
        join_df = event_df.join(dedup_df, on='event', how='left').fill_none('BLANK')
        
        # Round the statistics
        join_df = join_df.with_columns([
            pl.col('GA/Game').round(2).alias('GA/Game'),
            pl.col('GF/Game').round(2).alias('GF/Game')
        ])

        # Append results to the lists
        team_fixt_data.append(join_df[['next']].with_columns([pl.lit(team).alias('team')]))
        team_fdr_data.append(join_df[['next_fdr']].with_columns([pl.lit(team).alias('team')]))
        team_ga_data.append(join_df[['GA/Game']].with_columns([pl.lit(team).alias('team')]))
        team_gf_data.append(join_df[['GF/Game']].with_columns([pl.lit(team).alias('team')]))

    # Concatenate all the results once at the end to avoid repeated DataFrame concatenation
    team_fdr_df = pl.concat(team_fdr_data).set_index(0)
    team_fixt_df = pl.concat(team_fixt_data).set_index(0)
    team_ga_df = pl.concat(team_ga_data).set_index(0)
    team_gf_df = pl.concat(team_gf_data).set_index(0)
    
    return team_fdr_df, team_fixt_df, team_ga_df, team_gf_df


def get_current_season():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    start_year = events_df.iloc[0]['deadline_time'][:4]
    end_year = events_df.iloc[37]['deadline_time'][2:4]
    current_season = start_year + '/' + end_year
    return current_season
    

def get_player_url_list():
    id_dict = get_player_id_dict(order_by_col='id')
    url_list = [base_url + f'element-summary/{k}/' for k, v in id_dict.items()]
    return url_list


def filter_fixture_dfs_by_gw():
    fdr_df, fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
    ct_gw = get_current_gw()
    new_fixt_df = fixt_df.loc[:, ct_gw:(ct_gw+2)]
    new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
    new_fixt_df.columns = new_fixt_cols
    new_fdr_df = fdr_df.loc[:, ct_gw:(ct_gw+2)]
    new_fdr_df.columns = new_fixt_cols
    return new_fixt_df, new_fdr_df


def add_fixts_to_lg_table(new_fixt_df):
    league_df = get_league_table().join(new_fixt_df)
    league_df = league_df.reset_index()
    league_df.rename(columns={'team': 'Team'}, inplace=True)
    league_df.index += 1
    league_df['GD'] = league_df['GD'].map('{:+}'.format)
    return league_df


## Very slow to load, works but needs to be sped up.
def get_home_away_str_dict():
    new_fixt_df, new_fdr_df = filter_fixture_dfs_by_gw()
    result_dict = {}
    for column in new_fdr_df.columns:
        values = list(new_fdr_df[column])
        strings = list(new_fixt_df[column])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[column] = value_dict
    
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


def color_fixtures(val):
    ha_dict = get_home_away_str_dict()
    bg_color = 'background-color: '
    font_color = 'color: '
    if any(i in val for i in ha_dict[1]):
        bg_color += '#147d1b'
    elif any(i in val for i in ha_dict[2]):
        bg_color += '#00ff78'
    elif any(i in val for i in ha_dict[3]):
        bg_color += '#eceae6'
    elif any(i in val for i in ha_dict[4]):
        bg_color += '#ff0057'
        font_color += 'white'
    elif any(i in val for i in ha_dict[5]):
        bg_color += '#920947'
        font_color += 'white'
    else:
        bg_color += ''
    style = bg_color + '; ' + font_color
    return style

############