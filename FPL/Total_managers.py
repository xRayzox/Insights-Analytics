import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        response = requests.get(url)
        response.raise_for_status()
        print(manager_id)
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
    """Fetch data for all managers using concurrent threads."""
    manager_data = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_manager_details, team_id): team_id for team_id in range(1, total_players + 1)}
        for future in as_completed(futures):
            manager_data.append(future.result())
    return manager_data

def main():
    """Main function to fetch and save manager data."""
    total_players = get_bootstrap_data()['total_players']
    manager_data = fetch_all_managers(total_players)
    manager_df = pd.DataFrame(manager_data)
    manager_df.to_csv('./data/managers.csv', index=False, header=True)

if __name__ == "__main__":
    main()
