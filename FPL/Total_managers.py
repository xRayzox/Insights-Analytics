import asyncio
import aiohttp
import pandas as pd

base_url = 'https://fantasy.premierleague.com/api/'

async def get_bootstrap_data() -> dict:
    """Fetch general data from the FPL API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{base_url}bootstrap-static/') as resp:
            resp.raise_for_status() 
            return await resp.json() 

async def get_manager_details(session, manager_id):
    """Fetch manager details by ID asynchronously."""
    url = f'{base_url}entry/{manager_id}/'
    print(f"Fetching data for manager ID: {manager_id}")  # Print tracking message
    async with session.get(url) as resp:
        if resp.status == 200:
            data = await resp.json()
            return {
                'Manager': f"{data['player_first_name']} {data['player_last_name']}",
                'ID': data['id'],
                'Username': data.get('name', None)
            }
        else:
            print(f"Error fetching data for manager ID {manager_id}: {resp.status}")
            return {'Manager': None, 'ID': manager_id, 'Username': None}

async def fetch_all_managers(total_players):
    """Fetch data for all managers concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [get_manager_details(session, team_id) for team_id in range(1, 100 + 1)]
        manager_data = await asyncio.gather(*tasks)
    return pd.DataFrame(manager_data)

async def main():
    """Main function to fetch and save manager data."""
    total_players = (await get_bootstrap_data())['total_players']
    manager_df = await fetch_all_managers(total_players)
    manager_df.to_csv('./data/managers.csv', index=False, header=True)

if __name__ == "__main__":
    asyncio.run(main())