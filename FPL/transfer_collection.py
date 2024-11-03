import pandas as pd
from fpl_api_collection import (
    get_bootstrap_data, get_player_id_dict, get_player_data,
    get_total_fpl_players, remove_moved_players
)
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Cache API calls to minimize repeated fetches
@lru_cache(maxsize=100)
def cached_bootstrap_data():
    return get_bootstrap_data()

@lru_cache(maxsize=100)
def cached_player_id_dict():
    return get_player_id_dict(order_by_col='now_cost', web_name=False)

@lru_cache(maxsize=100)
def cached_total_fpl_players():
    return get_total_fpl_players()

def get_ele_df():
    print('-------------------SUI1---------')
    # Fetch the cached bootstrap data once
    bootstrap_data = cached_bootstrap_data()

    # Create DataFrame for players
    ele_df = remove_moved_players(pd.DataFrame(bootstrap_data['elements']))
    teams_df = pd.DataFrame(bootstrap_data['teams'])
    
    # Map the team names and create full name for players
    team_mapping = teams_df.set_index('id')['short_name'].to_dict()
    ele_df['team'] = ele_df['team'].map(team_mapping)
    ele_df['full_name'] = ele_df['first_name'] + ' ' + ele_df['second_name'] + ' (' + ele_df['team'] + ')'

    # Rename columns for readability and adjust price
    renaming_columns_and_price(ele_df)
    
    # Calculate transfer net and percentages
    total_players = cached_total_fpl_players()
    ele_df['%_+/-'] = (ele_df['T_+/-'] / total_players).astype(float)
    
    # Set the index and sort by transfer difference
    ele_df.set_index('Name', inplace=True)
    ele_df.sort_values('T_+/-', ascending=False, inplace=True)
    
    return ele_df

def renaming_columns_and_price(ele_df):
    rn_cols = {
        'web_name': 'Name', 'team': 'Team', 'element_type': 'Pos', 
        'event_points': 'GW_Pts', 'total_points': 'Pts', 'now_cost': 'Price',
        'selected_by_percent': 'TSB%', 'minutes': 'Mins', 'goals_scored': 'GS',
        'assists': 'A', 'penalties_missed': 'Pen_Miss', 'clean_sheets': 'CS',
        'goals_conceded': 'GC', 'own_goals': 'OG', 'penalties_saved': 'Pen_Save',
        'saves': 'S', 'yellow_cards': 'YC', 'red_cards': 'RC', 'bonus': 'B', 
        'bps': 'BPS', 'value_form': 'Value', 'points_per_game': 'PPG', 'influence': 'I',
        'creativity': 'C', 'threat': 'T', 'ict_index': 'ICT', 'influence_rank': 'I_Rank', 
        'creativity_rank': 'C_Rank', 'threat_rank': 'T_Rank', 'ict_index_rank': 'ICT_Rank',
        'transfers_in_event': 'T_In', 'transfers_out_event': 'T_Out', 
        'transfers_in': 'T_In_Total', 'transfers_out': 'T_Out_Total'
    }
    
    ele_df.rename(columns=rn_cols, inplace=True)
    ele_df['Price'] = ele_df['Price'] / 10
    ele_cols = ['Name', 'Team', 'Pos', 'Pts', 'Price', 'TSB%', 'T_In', 'T_Out', 
                'T_In_Total', 'T_Out_Total', 'full_name']
    ele_df = ele_df[ele_cols]
    
    # Calculate transfer net
    ele_df['T_+/-'] = ele_df['T_In'] - ele_df['T_Out']
    return ele_df

def collate_tran_df_from_name(ele_df, player_name):
    print('-------------------SUI2---------')
    # Filter player from elements DataFrame
    player_df = ele_df.loc[ele_df['full_name'] == player_name]
    
    # Get player ID from player dictionary
    full_player_dict = cached_player_id_dict()
    p_id = next((k for k, v in full_player_dict.items() if v == player_name), None)
    
    if p_id is None:  # Handle case where player is not found
        return pd.DataFrame()
    
    # Get player data from API
    p_data = get_player_data(str(p_id))
    
    if len(p_data['history']) == 0:  # Handle missing historical data
        return pd.DataFrame()
    
    # Create DataFrame for player's history and process data
    p_df = pd.DataFrame(p_data['history'])
    p_df = process_player_history(p_df, player_df)
    
    return p_df

def process_player_history(p_df, player_df):
    col_rn_dict = {'round': 'GW', 'value': 'Price', 'selected': 'SB', 
                   'transfers_in': 'Tran_In', 'transfers_out': 'Tran_Out'}
    p_df.rename(columns=col_rn_dict, inplace=True)
    p_df = p_df[['GW', 'Price', 'SB', 'Tran_In', 'Tran_Out']]
    p_df['Price'] = p_df['Price'] / 10
    
    # Append the latest week data
    new_df = pd.DataFrame({
        'GW': [p_df['GW'].max() + 1],
        'Price': [player_df['Price'].iloc[0]],
        'Tran_In': [player_df['T_In'].iloc[0]],
        'Tran_Out': [player_df['T_Out'].iloc[0]],
        'SB': [p_df['SB'].iloc[-1] + player_df['T_In'].iloc[0] - player_df['T_Out'].iloc[0]]
    })
    p_df = pd.concat([p_df, new_df], ignore_index=True)
    p_df.set_index('GW', inplace=True)
    
    return p_df

def get_hist_prices_df():
    print('-------------------SUI3---------')
    ele_df = get_ele_df()
    ordered_names = ele_df['full_name'].tolist()
    
    # Use ThreadPoolExecutor to parallelize player data fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda name: process_player(ele_df, name), ordered_names))
    
    df_list = [res for res in results if res is not None]
    
    if df_list:
        total_df = pd.concat(df_list, ignore_index=True)
        total_df.sort_values('+/-', ascending=False, inplace=True)
        total_df.set_index('Player', inplace=True)
        return total_df
    else:
        return pd.DataFrame()

def process_player(ele_df, name):
    print(name)
    p_hist_df = collate_tran_df_from_name(ele_df, name)
    if not p_hist_df.empty:
        sp = p_hist_df['Price'].iloc[0]
        np = p_hist_df['Price'].iloc[-1]
        return pd.DataFrame({'Player': [name], 'Start': [sp], 'Now': [np], '+/-': [np - sp]})
    else:
        return None

def write_data():
    print('-------------------SUI4---------')
    prices_df = get_hist_prices_df()
    if not prices_df.empty:
        prices_df['Start'] = prices_df['Start'].map('{:,.1f}'.format)
        prices_df['Now'] = prices_df['Now'].map('{:,.1f}'.format)
        prices_df['+/-'] = prices_df['+/-'].map('{:,.1f}'.format)
        # prices_df.to_csv('./data/player_prices.csv', index=True)
    return prices_df
