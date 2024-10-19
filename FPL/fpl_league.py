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

@lru_cache(maxsize=128)
def fetch_league_info(league_id):
    r: dict = requests.get(base_url + f"leagues-classic/{league_id}/standings/").json()
    if "league" not in r:
        r = requests.get(base_url + f"leagues-h2h/{league_id}/standings").json()
    if "league" not in r:
        raise ValueError(f"Could not find data for league_id: {league_id}")

    return {
        'entries': [entry_from_standings(e) for e in r['standings']['results']]
    }

@lru_cache(maxsize=128)
def get_manager_details(team_id):
    r = requests.get(base_url + f"entry/{team_id}/").json()
    return r

@lru_cache(maxsize=128)
def get_manager_history_data(team_id):
    r = requests.get(base_url + f"entry/{team_id}/history/").json()
    return r

@lru_cache(maxsize=128)
def get_bootstrap_data():
    r = requests.get(base_url + "bootstrap-static/").json()
    return r
