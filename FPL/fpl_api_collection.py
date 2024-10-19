import pandas as pd
import requests
from dataclasses import dataclass
from functools import lru_cache

base_url = 'https://fantasy.premierleague.com/api/'
@dataclass
class Entry:
    team_id: int
    name: str
    player_name: str
    rank: int

def entry_from_standings(standings) -> Entry:
    return Entry(
        team_id=standings['entry'],
        name=standings['entry_name'],
        player_name=standings["player_name"],
        rank=standings['rank']
    )
@dataclass
class LeagueInfo:
    id: int
    name: str
    entries: list[Entry]
# Function to get general data (bootstrap data) from the FPL API
@lru_cache(maxsize=128)
def get_bootstrap_data() -> dict:
    """
    Options
    -------
        ['element_stats']
        ['element_types']
        ['elements']
        ['events']
        ['game_settings']
        ['phases']
        ['teams']
        ['total_players']
    """
    resp = requests.get(f'{base_url}bootstrap-static/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

# Function to get fixture data (upcoming matches)
@lru_cache(maxsize=128)
def get_fixture_data() -> dict:
    resp = requests.get(f'{base_url}fixtures/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

# Function to get player-specific data
def get_player_data(player_id) -> dict:
    """
    Options
    -------
        ['fixtures']
        ['history']
        ['history_past']
    """
    resp = requests.get(f'{base_url}element-summary/{player_id}/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

# Function to get FPL manager details
@lru_cache(maxsize=128)
def get_manager_details(manager_id) -> dict:
    resp = requests.get(f'{base_url}entry/{manager_id}/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

# Function to get FPL manager's history (past seasons' performances)
@lru_cache(maxsize=128)
def get_manager_history_data(manager_id) -> dict:
    resp = requests.get(f'{base_url}entry/{manager_id}/history/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

# Function to get a manager's selected team for a given gameweek (GW)
@lru_cache(maxsize=128)
def get_manager_team_data(manager_id, gw):
    """
    Options
    -------
        ['active_chip']
        ['automatic_subs']
        ['entry_history']
        ['picks']
    """
    resp = requests.get(f'{base_url}entry/{manager_id}/event/{gw}/picks/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    return resp.json()

# Function to get the total number of FPL players (managers)
def get_total_fpl_players():
    return get_bootstrap_data()['total_players']



# Function to filter out players who have left their clubs or are on loan
def remove_moved_players(df):
    strings = ['loan', 'Loan', 'Contract cancelled', 'Left the club',
               'Permanent', 'Released', 'Signed for', 'Transferred',
               'Season long', 'Not training', 'permanent', 'transferred']
    df_copy = df.loc[~df['news'].str.contains('|'.join(strings), case=False)]
    return df_copy

# Function to create a dictionary mapping player IDs to their names
def get_player_id_dict(order_by_col, web_name=True) -> dict:
    ele_df = pd.DataFrame(get_bootstrap_data()['elements'])
    ele_df = remove_moved_players(ele_df)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    ele_df['team_name'] = ele_df['team'].map(teams_df.set_index('id')['short_name'])
    ele_df.sort_values(order_by_col, ascending=False, inplace=True)
    if web_name == True:
        id_dict = dict(zip(ele_df['id'], ele_df['web_name']))
    else:
        ele_df['full_name'] = ele_df['first_name'] + ' ' + \
            ele_df['second_name'] + ' (' + ele_df['team_name'] + ')'
        id_dict = dict(zip(ele_df['id'], ele_df['full_name']))
    return id_dict

# Function to gather historic gameweek data for all players
@lru_cache(maxsize=128)
def collate_player_hist():
    res = []
    p_dict = get_player_id_dict()
    for p_id, p_name in p_dict.items():
        resp = requests.get('{}element-summary/{}/'.format(base_url, p_id))
        if resp.status_code != 200:
            print('Request to {} data failed'.format(p_name))
            raise Exception(f'Response was status code {resp.status_code}')
        else:
            res.append(resp.json()['history'])
    return pd.DataFrame(res)


# Team, games_played, wins, losses, draws, goals_for, goals_against, GD,
# PTS, Form? [W,W,L,D,W]

# Function to create a league table based on fixture results
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
        # df = df.loc[df['finished'] == True]
        df.sort_values('event', inplace=True)
        df.loc[(df['was_home'] == True) &
               (df['team_h_score'] > df['team_a_score']), 'win'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_a_score'] > df['team_h_score']), 'win'] = True
        df.loc[(df['team_h_score'] == df['team_a_score']), 'draw'] = True
        df.loc[(df['was_home'] == True) &
               (df['team_h_score'] < df['team_a_score']), 'loss'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_a_score'] < df['team_h_score']), 'loss'] = True
        df.loc[(df['was_home'] == True), 'gf'] = df['team_h_score']
        df.loc[(df['was_home'] == False), 'gf'] = df['team_a_score']
        df.loc[(df['was_home'] == True), 'ga'] = df['team_a_score']
        df.loc[(df['was_home'] == False), 'ga'] = df['team_h_score']
        df.loc[(df['win'] == True), 'result'] = 'W'
        df.loc[(df['draw'] == True), 'result'] = 'D'
        df.loc[(df['loss'] == True), 'result'] = 'L'
        df.loc[(df['was_home'] == True) &
               (df['team_a_score'] == 0), 'clean_sheet'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_h_score'] == 0), 'clean_sheet'] = True
        ws = len(df.loc[df['win'] == True])
        ds = len(df.loc[df['draw'] == True])
        finished_df = df.loc[df['finished'] == True]
        l_data = {'id': [t_id], 'GP': [len(finished_df)], 'W': [ws], 'D': [ds],
                  'L': [len(df.loc[df['loss'] == True])],
                  'GF': [df['gf'].sum()], 'GA': [df['ga'].sum()],
                  'GD': [df['gf'].sum() - df['ga'].sum()],
                  'CS': [df['clean_sheet'].sum()], 'Pts': [(ws*3) + ds],
                  'Form': [finished_df['result'].tail(5).str.cat(sep='')]}
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

    league_df['Pts/Game'] = (league_df['Pts']/league_df['GP']).round(2)
    league_df['GF/Game'] = (league_df['GF']/league_df['GP']).round(2)
    league_df['GA/Game'] = (league_df['GA']/league_df['GP']).round(2)
    league_df['CS/Game'] = (league_df['CS']/league_df['GP']).round(2)
    
    return league_df


def get_current_gw():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    current_gw = events_df.loc[events_df['is_next'] == True].reset_index()['id'][0]
    return current_gw


def get_current_season():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    id_first = events_df['deadline_time'].str[:4].iloc[0]
    id_last = events_df['deadline_time'].str[2:4].iloc[-1]
    current_season = str(id_first) + '/' + str(id_last)
    return current_season


def get_fixture_dfs():
    # doubles??
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().tolist()
    # don't need to worry about double fixtures just yet!
    fixt_df['team_h'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['team_a'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    gw_dict = dict(zip(range(1,381),
                       [num for num in range(1, 39) for x in range(10)]))
    fixt_df['event_lock'] = fixt_df['id'].map(gw_dict)
    team_fdr_data = []
    team_fixt_data = []
    for team in teams_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == team]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == team]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values('event_lock', inplace=True)
        h_filt = (df['team_h'] == team) & (df['event'].notnull())
        a_filt = (df['team_a'] == team) & (df['event'].notnull())
        df.loc[h_filt, 'next'] = df['team_a'] + ' (H)'
        df.loc[a_filt, 'next'] = df['team_h'] + ' (A)'
        df.loc[df['event'].isnull(), 'next'] = 'BLANK'
        df.loc[h_filt, 'next_fdr'] = df['team_h_difficulty']
        df.loc[a_filt, 'next_fdr'] = df['team_a_difficulty']
        team_fixt_data.append(pd.DataFrame([team] + list(df['next'])).transpose())
        team_fdr_data.append(pd.DataFrame([team] + list(df['next_fdr'])).transpose())
    team_fdr_df = pd.concat(team_fdr_data).set_index(0)
    team_fixt_df = pd.concat(team_fixt_data).set_index(0)
    return team_fdr_df, team_fixt_df


def get_fixt_dfs():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().tolist()
    league_df = get_league_table().reset_index()
    fixt_df['team_h'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['team_a'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    
    gw_dict = dict(zip(range(1,381),
                       [num for num in range(1, 39) for x in range(10)]))
    fixt_df['event_lock'] = fixt_df['id'].map(gw_dict)
    team_fdr_data = []
    team_fixt_data = []
    team_ga_data = []
    team_gf_data = []
    for team in teams_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == team]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == team]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values(['kickoff_time'], inplace=True)
        h_filt = (df['team_h'] == team) & (df['event'].notnull())
        a_filt = (df['team_a'] == team) & (df['event'].notnull())
        df.loc[h_filt, 'next'] = df['team_a'] + ' (H)'
        df.loc[a_filt, 'next'] = df['team_h'] + ' (A)'
        df['team'] = df['next'].str[:3]
        dup_df = df.duplicated(subset=['event'], keep=False).reset_index()
        dup_df.columns = ['index', 'multiple']
        df = df.reset_index().merge(dup_df, on='index', how='left')
        df.set_index('index', inplace=True)
        df.loc[h_filt, 'next_fdr'] = df['team_h_difficulty']
        df.loc[a_filt, 'next_fdr'] = df['team_a_difficulty']
        new_df = df.merge(league_df[['team', 'GA/Game', 'GF/Game']], on='team', how='left')
        event_df = pd.DataFrame({'event': [num for num in range(1, 39)]})
        dedup_df = df.groupby('event').agg({'next': ' + '.join}).reset_index()
        dedup_fdr_df = new_df.groupby('event')[['next_fdr', 'GA/Game', 'GF/Game']].mean().reset_index()
        dedup_df = dedup_df.merge(dedup_fdr_df, on='event', how='left')
        join_df = event_df.merge(dedup_df, on='event', how='left')
        join_df.loc[join_df['next'].isnull(), 'next'] = 'BLANK'
        join_df['GA/Game'] = join_df['GA/Game'].apply(lambda x: round(x, 2))
        join_df['GF/Game'] = join_df['GF/Game'].apply(lambda x: round(x, 2))
        team_fixt_data.append(pd.DataFrame([team] + list(join_df['next'])).transpose())
        team_fdr_data.append(pd.DataFrame([team] + list(join_df['next_fdr'])).transpose())
        team_ga_data.append(pd.DataFrame([team] + list(join_df['GA/Game'])).transpose())
        team_gf_data.append(pd.DataFrame([team] + list(join_df['GF/Game'])).transpose())
    team_fdr_df = pd.concat(team_fdr_data).set_index(0)
    team_fixt_df = pd.concat(team_fixt_data).set_index(0)
    team_ga_df = pd.concat(team_ga_data).set_index(0)
    team_gf_df = pd.concat(team_gf_data).set_index(0)
    return team_fdr_df, team_fixt_df, team_ga_df, team_gf_df


def get_current_season():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    start_year = events_df.iloc[0]['deadline_time'][:4]
    end_year = events_df.iloc[37]['deadline_time'][2:4]
    current_season = start_year + '/' + end_year
    return current_season
    
@lru_cache(maxsize=128)
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
@lru_cache(maxsize=128)
def fetch_league_info(league_id: int) -> LeagueInfo:
    r: dict = requests.get(base_url + f"leagues-classic/{league_id}/standings/").json()
    if "league" not in r:
        r = requests.get(base_url + f"leagues-h2h/{league_id}/standings").json()
    if "league" not in r:
        raise ValueError(f"Could not find data for league_id: {league_id}")
    return LeagueInfo(
        id=r["league"]['id'],
        name=r["league"]["name"],
        entries=[entry_from_standings(e) for e in r['standings']['results']]
    )