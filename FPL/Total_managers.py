import aiohttp
import asyncio
import pandas as pd
import time
from aiohttp import ClientSession, ClientError

base_url = 'https://fantasy.premierleague.com/api/'

# Set a max retry limit and delay between retries for error handling
MAX_RETRIES = 3
RETRY_DELAY = 1  # In seconds
BATCH_SIZE = 1000  # Number of managers to fetch in one batch

async def fetch_bootstrap_data(session: ClientSession):
    """Fetch general data from the FPL API asynchronously."""
    async with session.get(f'{base_url}bootstrap-static/') as response:
        response.raise_for_status()  # Will raise an HTTPError if response code isn't 200
        return await response.json()

async def get_manager_details(session: ClientSession, manager_id: int, retries: int = 0):
    """Fetch manager details by ID with retry on failure."""
    url = f'{base_url}entry/{manager_id}/'
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()  # Check if the request was successful
            data = await response.json()
            return {
                'Manager': f"{data['player_first_name']} {data['player_last_name']}",
                'ID': data['id'],
                'Username': data.get('name', None)
            }
    except (ClientError, asyncio.TimeoutError) as e:
        if retries < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
            return await get_manager_details(session, manager_id, retries + 1)  # Retry recursively
        print(f"Error fetching data for manager ID {manager_id}: {e}")
        return {'Manager': None, 'ID': manager_id, 'Username': None}

async def fetch_batch_managers(session: ClientSession, start_id: int, end_id: int):
    """Fetch data for a batch of managers asynchronously."""
    tasks = [get_manager_details(session, manager_id) for manager_id in range(start_id, end_id + 1)]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result]  # Filter out None results

async def fetch_all_managers(total_players: int):
    """Fetch data for all managers using batching and concurrent requests."""
    try:
        history_manager = pd.read_csv('./data/managers.csv')
        current_count = len(history_manager) + 1
    except FileNotFoundError:
        history_manager = pd.DataFrame()  # If the file doesn't exist, start with an empty DataFrame
        current_count = 1

    # Fetch missing manager data in batches
    manager_data = []
    async with aiohttp.ClientSession() as session:
        batch_start = current_count
        batch_end = min(batch_start + BATCH_SIZE - 1, total_players)

        while batch_start <= total_players:
            print(f"Fetching managers {batch_start} to {batch_end}...")
            batch_results = await fetch_batch_managers(session, batch_start, batch_end)
            manager_data.extend(batch_results)

            # Update the batch range
            batch_start += BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE - 1, total_players)

    # Append new data to existing DataFrame
    new_data_df = pd.DataFrame(manager_data)
    history_manager = pd.concat([history_manager, new_data_df], ignore_index=True)

    return history_manager

async def main():
    """Main function to fetch and save manager data."""
    try:
        async with aiohttp.ClientSession() as session:
            total_players_data = await fetch_bootstrap_data(session)
            total_players = total_players_data['total_players']
            print(f"Fetching data for {total_players} managers...")

            # Fetch and append new manager data
            manager_data = await fetch_all_managers(total_players)

            # Save the updated manager data to CSV
            manager_data.to_csv('./data/managers.csv', index=False, header=True)
            print("Manager data successfully saved to './data/managers.csv'.")
    except (ClientError, asyncio.TimeoutError) as e:
        print(f"An error occurred during the process: {e}")

# Run the async main function
if __name__ == "__main__":
    start_time = time.time()  # Track the start time for performance measurement
    asyncio.run(main())
    print(f"Execution time: {time.time() - start_time} seconds")
