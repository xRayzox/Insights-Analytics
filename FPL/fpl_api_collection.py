import pandas as pd
import aiohttp
import asyncio
import requests
from cachetools import TTLCache

base_url = 'https://fantasy.premierleague.com/api/'
cache = TTLCache(maxsize=100, ttl=600)  # Cache for 10 minutes

async def fetch(url, session):
    async with session.get(url) as resp:
        if resp.status != 200:
            raise Exception(f'Response was status code {resp.status}')
        return await resp.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(url, session) for url in urls]
        return await asyncio.gather(*tasks)

def get_bootstrap_data():
    if 'bootstrap_data' in cache:
        return cache['bootstrap_data']
    url = f'{base_url}bootstrap-static/'
    data = requests.get(url).json()
    cache['bootstrap_data'] = data
    return data

def get_fixture_data():
    if 'fixture_data' in cache:
        return cache['fixture_data']
    url = f'{base_url}fixtures/'
    data = requests.get(url).json()
    cache['fixture_data'] = data
    return data

def get_player_data(player_id):
    url = f'{base_url}element-summary/{player_id}/'
    return requests.get(url).json()

async def get_all_players_data():
    player_ids = [player['id'] for player in get_bootstrap_data()['elements']]
    urls = [f'{base_url}element-summary/{player_id}/' for player_id in player_ids]
    return await fetch_all(urls)

def get_manager_details(manager_id):
    url = f'{base_url}entry/{manager_id}/'
    return requests.get(url).json()

def get_manager_history_data(manager_id):
    url = f'{base_url}entry/{manager_id}/history/'
    return requests.get(url).json()

def get_manager_team_data(manager_id, gw):
    url = f'{base_url}entry/{manager_id}/event/{gw}/picks/'
    return requests.get(url).json()

def get_total_fpl_players():
    return get_bootstrap_data()['total_players']

def remove_moved_players(df):
    strings = ['loan', 'Contract cancelled', 'Left the club', 'Permanent', 'Released']
    return df[~df['news'].str.contains('|'.join(strings), case=False)]

def get_player_id_dict(order_by_col='id', web_name=True):
    ele_df = pd.DataFrame(get_bootstrap_data()['elements'])
    ele_df = remove_moved_players(ele_df)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    ele_df['team_name'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])
    ele_df.sort_values(order_by_col, ascending=False, inplace=True)
    if web_name:
        return dict(zip(ele_df['id'], ele_df['web_name']))
    else:
        ele_df['full_name'] = ele_df['first_name'] + ' ' + ele_df['second_name'] + ' (' + ele_df['team_name'] + ')'
        return dict(zip(ele_df['id'], ele_df['full_name']))

def collate_player_hist():
    res = []
    player_data = asyncio.run(get_all_players_data())
    for player_history in player_data:
        res.append(player_history['history'])
    return pd.DataFrame(res)

def get_league_table():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_id_list = teams_df['id'].unique().tolist()
    df_list = []
    for t_id in teams_id_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == t_id]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == t_id]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values('event', inplace=True)
        df.loc[(df['was_home'] == True) & (df['team_h_score'] > df['team_a_score']), 'win'] = True
        df.loc[(df['was_home'] == False) & (df['team_a_score'] > df['team_h_score']), 'win'] = True
        df.loc[(df['team_h_score'] == df['team_a_score']), 'draw'] = True
        df.loc[(df['was_home'] == True) & (df['team_h_score'] < df['team_a_score']), 'loss'] = True
        df.loc[(df['was_home'] == False) & (df['team_a_score'] < df['team_h_score']), 'loss'] = True
        df.loc[(df['was_home'] == True), 'gf'] = df['team_h_score']
        df.loc[(df['was_home'] == False), 'gf'] = df['team_a_score']
        df.loc[(df['was_home'] == True), 'ga'] = df['team_a_score']
        df.loc[(df['was_home'] == False), 'ga'] = df['team_h_score']
        df.loc[(df['win'] == True), 'result'] = 'W'
        df.loc[(df['draw'] == True), 'result'] = 'D'
        df.loc[(df['loss'] == True), 'result'] = 'L'
        df.loc[(df['was_home'] == True) & (df['team_a_score'] == 0), 'clean_sheet'] = True
        df.loc[(df['was_home'] == False) & (df['team_h_score'] == 0), 'clean_sheet'] = True
        ws = len(df.loc[df['win'] == True])
        ds = len(df.loc[df['draw'] == True])
        finished_df = df.loc[df['finished'] == True]
        l_data = {
            'id': [t_id],
            'GP': [len(finished_df)],
            'W': [ws],
            'D': [ds],
            'L': [len(df.loc[df['loss'] == True])],
            'GF': [df['gf'].sum()],
            'GA': [df['ga'].sum()],
            'GD': [df['gf'].sum() - df['ga'].sum()],
            'CS': [df['clean_sheet'].sum()],
            'Pts': [(ws*3) + ds],
            'Form': [finished_df['result'].tail(5).str.cat(sep='')]
        }
        df_list.append(pd.DataFrame(l_data))
    league_df = pd.concat(df_list)
    league_df['team'] = league_df['id'].map(teams_df.set_index('id')['short_name'])
    league_df.drop('id', axis=1, inplace=True)
    league_df.reset_index(drop=True, inplace=True)
    league_df.sort_values(['Pts', 'GD', 'GF', 'GA'], ascending=False, inplace=True)
    league_df.set_index('team', inplace=True)
    league_df['GF'] = league_df['GF'].astype(int)
    league_df['GA'] = league_df['GA'].astype(int)
    league_df['GD'] = league_df['GD'].astype(int)

    league_df['Pts/Game'] = (league_df['Pts'] / league_df['GP']).round(2)
    league_df['GF/Game'] = (league_df['GF'] / league_df['GP']).round(2)
    league_df['GA/Game'] = (league_df['GA'] / league_df['GP']).round(2)
    league_df['CS/Game'] = (league_df['CS'] / league_df['GP']).round(2)
    
    return league_df

def get_current_gw():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    current_gw = events_df.loc[events_df['is_next'] == True].reset_index()['id'][0]
    return current_gw

def get_current_season():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    start_year = events_df.iloc[0]['deadline_time'][:4]
    end_year = events_df.iloc[37]['deadline_time'][2:4]
    current_season = start_year + '/' + end_year
    return current_season

def get_fixture_dfs():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().tolist()
    fixt_df['team_h'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['team_a'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    
    gw_dict = dict(zip(range(1, 381), [num for num in range(1, 39) for x in range(10)]))
    fixt_df['event_lock'] = fixt_df['id'].map(gw_dict)
    team_fdr_data = []
    team_fixt_data = []

    for team in teams_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == team]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == team]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values('event', inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        fdr = []
        for x in range(len(df)):
            if df.iloc[x]['was_home']:
                fdr.append(df.iloc[x]['team_a'])
            else:
                fdr.append(df.iloc[x]['team_h'])
        df['FDR'] = pd.Series(fdr)
        team_fdr_data.append(df[['event', 'date', 'FDR']])
        team_fixt_data.append(df[['event', 'date', 'team_h', 'team_a', 'team_h_score', 'team_a_score']])
    team_fdr_df = pd.concat(team_fdr_data).reset_index(drop=True)
    team_fixt_df = pd.concat(team_fixt_data).reset_index(drop=True)
    return team_fdr_df, team_fixt_df

