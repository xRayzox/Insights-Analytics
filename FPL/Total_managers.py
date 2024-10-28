import requests
import pandas as pd
import dask.dataframe as dd
from dask import delayed

base_url = 'https://fantasy.premierleague.com/api/'

def get_bootstrap_data():
    """Fetch general data from the FPL API."""
    response = requests.get(f'{base_url}bootstrap-static/')
    response.raise_for_status()
    return response.json()

def get_manager_details(manager_id):
    """Fetch manager details by ID."""
    url = f'{base_url}entry/{manager_id}/'
    try:
        print(f"Fetching data for manager ID: {manager_id}")  # Print tracking message
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'Manager': f"{data['player_first_name']} {data['player_last_name']}",
            'ID': data['id'],
            'Username': data.get('name', None)
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data for manager ID {manager_id}: {e}")
        return {'Manager': None, 'ID': manager_id, 'Username': None}

def fetch_all_managers(total_players):
    """Fetch data for all managers using Dask for parallelization."""
    tasks = [delayed(get_manager_details)(team_id) for team_id in range(1, total_players + 1)]
    manager_data = dd.from_delayed(tasks)
    return manager_data

def main():
    """Main function to fetch and save manager data."""
    total_players = get_bootstrap_data()['total_players']
    manager_ddf = fetch_all_managers(total_players)
    manager_ddf.compute().to_csv('./data/managers.csv', index=False, header=True)

if __name__ == "__main__":
    main()
