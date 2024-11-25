import aiohttp
import asyncio
import pandas as pd
import time
import nest_asyncio
import os

# Apply nest_asyncio to allow nested event loops in Colab
nest_asyncio.apply()

base_url = 'https://fantasy.premierleague.com/api/'

# Set parameters for error handling and performance tuning
MAX_RETRIES = 3
RETRY_DELAY = 1  # In seconds
BATCH_SIZE = 300  # Increased for faster fetching
TIMEOUT = 3  # Timeout for each request
CONCURRENT_LIMIT = 100  # Maximum number of concurrent connections

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

async def fetch_bootstrap_data(session):
    """Fetch general data from the FPL API asynchronously."""
    async with session.get(f'{base_url}bootstrap-static/', timeout=TIMEOUT) as response:
        response.raise_for_status()
        return await response.json()

async def get_manager_details(session, manager_id, retries=0):
    """Fetch manager details by ID with retry on failure."""
    url = f'{base_url}entry/{manager_id}/'
    try:
        async with session.get(url, timeout=TIMEOUT) as response:
            response.raise_for_status()
            data = await response.json()
            return {
                'Manager': f"{data['player_first_name']} {data['player_last_name']}",
                'ID': data['id'],
                'Username': data.get('name', None)
            }
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        if retries < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)
            return await get_manager_details(session, manager_id, retries + 1)
        print(f"Error fetching data for manager ID {manager_id}: {e}")
        return {'Manager': None, 'ID': manager_id, 'Username': None}

async def fetch_batch_managers(session, start_id, end_id):
    """Fetch data for a batch of managers asynchronously."""
    tasks = [
        get_manager_details(session, manager_id)
        for manager_id in range(start_id, end_id + 1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return [result for result in results if result]

async def fetch_all_managers(total_players):
    """Fetch data for all managers using batching and concurrent requests."""
    try:
        # File names for storing data in the 'data' folder
        file_names = [
            "data/clean_Managers_part1.csv", "data/clean_Managers_part2.csv", "data/clean_Managers_part3.csv",
            "data/clean_Managers_part4.csv", "data/clean_Managers_part5.csv"
        ]

        dataframes = [pd.read_csv(file,low_memory=False) for file in file_names if os.path.exists(file)]
        history_manager = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        current_count = len(history_manager) + 1
    except FileNotFoundError:
        history_manager = pd.DataFrame()
        current_count = 1

    connector = aiohttp.TCPConnector(limit=CONCURRENT_LIMIT)
    async with aiohttp.ClientSession(connector=connector) as session:
        batch_start = current_count
        batch_end = min(batch_start + BATCH_SIZE - 1, total_players)
        
        file_index = 0  # To determine which file to append to (out of 5)
        
        while batch_start <= total_players:
            print(f"Fetching managers {batch_start} to {batch_end}...")
            batch_results = await fetch_batch_managers(session, batch_start, batch_end)
            manager_data = pd.DataFrame(batch_results)
            
            # Append data to the appropriate CSV file
            current_file = file_names[file_index]
            manager_data.to_csv(
                current_file,
                mode='a',
                index=False,
                header=not os.path.exists(current_file)
            )
            print(f"Batch {batch_start} to {batch_end} appended to {current_file}.")

            # Update the batch range
            batch_start += BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE - 1, total_players)
            
            # Update the file index to cycle through the five files
            file_index = (file_index + 1) % len(file_names)

    return history_manager

async def fetch_missing_manager_details(session, missing_ids):
    """Fetch data for managers with missing details."""
    tasks = [
        get_manager_details(session, manager_id)
        for manager_id in missing_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return [result for result in results if result]

async def update_missing_values_in_file(file_path):
    """Identify and fetch missing manager data in a specific file."""
    # Read the file into a DataFrame
    try:
        history_manager = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Identify rows with missing values
    rows_with_missing = history_manager[history_manager.isnull().any(axis=1)]
    
    # Get the list of IDs with missing values
    ids_with_missing_values = rows_with_missing['ID'].unique()
    

    # Fetch missing manager details
    connector = aiohttp.TCPConnector(limit=CONCURRENT_LIMIT)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Fetching details for {len(ids_with_missing_values)} missing IDs in {file_path}...")
        missing_manager_data = await fetch_missing_manager_details(session, ids_with_missing_values)

        # Convert to dataframe
        missing_data_df = pd.DataFrame(missing_manager_data)

        # Update the history_manager DataFrame with the missing data
        for index, row in missing_data_df.iterrows():
            manager_id = row['ID']
            missing_row = history_manager[history_manager['ID'] == manager_id]
            if not missing_row.empty:
                history_manager.loc[history_manager['ID'] == manager_id, ['Manager', 'Username']] = row[['Manager', 'Username']].values

        # Save the updated DataFrame back to the CSV
        history_manager.to_csv(file_path, index=False)
        print(f"Missing values updated and saved in {file_path}.")

async def update_missing_values():
    """Check and update missing values for each CSV file."""
    file_names = [
        "data/clean_Managers_part1.csv", "data/clean_Managers_part2.csv", "data/clean_Managers_part3.csv",
        "data/clean_Managers_part4.csv", "data/clean_Managers_part5.csv"
    ]
    
    for file_path in file_names:
        await update_missing_values_in_file(file_path)

async def main():
    """Main function to fetch, update and save manager data."""
    try:
        async with aiohttp.ClientSession() as session:
            total_players_data = await fetch_bootstrap_data(session)
            total_players = total_players_data['total_players']
            print(f"Fetching data for {total_players} managers...")

            # Fetch all manager data in batches
            await fetch_all_managers(total_players)

            # After fetching, check and update missing values
            await update_missing_values()
            print("All missing manager data has been updated.")

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"An error occurred during the process: {e}")

# Run the async main function
start_time = time.time()
asyncio.run(main())
print(f"Execution time: {time.time() - start_time} seconds")
