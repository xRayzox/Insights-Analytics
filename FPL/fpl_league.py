import requests
from cachetools import TTLCache, cached
from cachetools.func import ttl_cache

# Base URL for FPL API
base_url = 'https://fantasy.premierleague.com/api/'

# Define cache TTL and max size
cache_maxsize = 128  # Maximum number of cache entries
cache_ttl = 120  # Time-to-Live in seconds

def entry_from_standings(standings):
    """Extract relevant information from league standings."""
    return {
        'team_id': standings['entry'],
        'name': standings['entry_name'],
        'player_name': standings["player_name"],
        'rank': standings['rank']
    }

@ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def fetch_league_info(league_id):
    """Fetch league standings, either classic or head-to-head."""
    r = requests.get(base_url + f"leagues-classic/{league_id}/standings/").json()
    if "league" not in r:
        r = requests.get(base_url + f"leagues-h2h/{league_id}/standings").json()
    if "league" not in r:
        raise ValueError(f"Could not find data for league_id: {league_id}")

    return {
        'entries': [entry_from_standings(e) for e in r['standings']['results']]
    }

@ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def get_manager_details(team_id):
    """Fetch detailed information about a manager's team."""
    r = requests.get(base_url + f"entry/{team_id}/").json()
    return r

@ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def get_manager_history_data(team_id):
    """Fetch historical data about a manager's team."""
    r = requests.get(base_url + f"entry/{team_id}/history/").json()
    return r

@ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def get_bootstrap_data():
    """Fetch static data that includes all players and game information."""
    r = requests.get(base_url + "bootstrap-static/").json()
    return r