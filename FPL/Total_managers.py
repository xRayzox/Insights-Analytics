import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

base_url = 'https://fantasy.premierleague.com/api/'

# Set a max retry limit and delay between retries for error handling
MAX_RETRIES = 3
RETRY_DELAY = 1  # In seconds

def get_bootstrap_data():
    """Fetch general data from the FPL API."""
    response = requests.get(f'{base_url}bootstrap-static/')
    response.raise_for_status()  # Will raise an HTTPError if response code isn't 200
    return response.json()

def get_manager_details(manager_id, retries=0):
    """Fetch manager details by ID with retry on failure."""
    url = f'{base_url}entry/{manager_id}/'
    try:
        response = requests.get(url, timeout=10)  # Set a timeout for the request
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        return {
            'Manager': f"{data['player_first_name']} {data['player_last_name']}",
            'ID': data['id'],
            'Username': data.get('name', None)
        }
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            time.sleep(RETRY_DELAY)  # Wait before retrying
            return get_manager_details(manager_id, retries + 1)  # Retry recursively
        print(f"Error fetching data for manager ID {manager_id}: {e}")
        return {'Manager': None, 'ID': manager_id, 'Username': None}

def fetch_all_managers(total_players, max_workers=10):
    """Fetch data for all managers using concurrent threads with a controlled pool."""
    # Read the current manager data if exists
    try:
        history_manager = pd.read_csv('./data/managers.csv')
        current_count = len(history_manager)+1
    except FileNotFoundError:
        history_manager = pd.DataFrame()  # If the file doesn't exist, start with an empty DataFrame
        current_count = 1
    


    # Fetch only missing manager data
    manager_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_manager_details, team_id): team_id for team_id in range(current_count, total_players+1)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                manager_data.append(result)

    # Append new data to existing DataFrame
    new_data_df = pd.DataFrame(manager_data)
    history_manager = pd.concat([history_manager, new_data_df], ignore_index=True)
    
    return history_manager

def main():
    """Main function to fetch and save manager data."""
    try:
        total_players = get_bootstrap_data()['total_players']
        print(f"Fetching data for {total_players} managers...")

        # Fetch and append new manager data
        manager_data = fetch_all_managers(total_players)
        
        # Save the updated manager data to CSV
        manager_data.to_csv('./data/managers.csv', index=False, header=True)
        print("Manager data successfully saved to './data/managers.csv'.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()
