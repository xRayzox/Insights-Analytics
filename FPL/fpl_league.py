import requests
from functools import lru_cache
import time

base_url = 'https://fantasy.premierleague.com/api/'

# Cache dictionary to manually manage TTL
cache = {}

def cache_with_ttl(key, ttl=120):
    current_time = time.time()
    if key in cache:
        value, timestamp = cache[key]
        if current_time - timestamp < ttl:
            return value
    return None

def set_cache(key, value):
    cache[key] = (value, time.time())

def entry_from_standings(standings):
    return {
        'team_id': standings['entry'],
        'name': standings['entry_name'],
        'player_name': standings["player_name"],
        'rank': standings['rank']
    }

def fetch_league_info(league_id):
    cache_key = f'league_info_{league_id}'
    cached_value = cache_with_ttl(cache_key)
    if cached_value is not None:
        return cached_value

    # Make API request
    r = requests.get(base_url + f"leagues-classic/{league_id}/standings/").json()
    if "league" not in r:
        r = requests.get(base_url + f"leagues-h2h/{league_id}/standings").json()
    if "league" not in r:
        raise ValueError(f"Could not find data for league_id: {league_id}")

    league_data = {
        'entries': [entry_from_standings(e) for e in r['standings']['results']]
    }
    
    set_cache(cache_key, league_data)
    return league_data

def get_manager_details(team_id):
    cache_key = f'manager_details_{team_id}'
    cached_value = cache_with_ttl(cache_key)
    if cached_value is not None:
        return cached_value

    r = requests.get(base_url + f"entry/{team_id}/").json()
    set_cache(cache_key, r)
    return r

def get_manager_history_data(team_id):
    cache_key = f'manager_history_{team_id}'
    cached_value = cache_with_ttl(cache_key)
    if cached_value is not None:
        return cached_value

    r = requests.get(base_url + f"entry/{team_id}/history/").json()
    set_cache(cache_key, r)
    return r

def get_bootstrap_data():
    cache_key = 'bootstrap_data'
    cached_value = cache_with_ttl(cache_key)
    if cached_value is not None:
        return cached_value

    r = requests.get(base_url + "bootstrap-static/").json()
    set_cache(cache_key, r)
    return r
