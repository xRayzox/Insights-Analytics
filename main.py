import time
import pandas as pd
import duckdb
import dask.dataframe as dd
import modin.pandas as mpd
import pyarrow.parquet as pq
import pyarrow.csv as pv_csv
import glob
import sqlite3

# 1. DuckDB - Optimal for Large Files
start_duckdb = time.time()
history_manager_duckdb = duckdb.query("""
    SELECT * FROM read_csv_auto('./data/manager/clean_Managers_part*.csv', header=True)
""").df()  # using `.df()` to get the result as a pandas DataFrame
end_duckdb = time.time()
print("DuckDB Load Time:", end_duckdb - start_duckdb)

# 2. Pandas (Efficient for Moderate Files)
start_pandas = time.time()
file_pattern = './data/manager/clean_Managers_part*.csv'
files = glob.glob(file_pattern)
history_manager_pandas = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
end_pandas = time.time()
print("Pandas Load Time:", end_pandas - start_pandas)

# 3. Dask (Parallel CSV Loading for Large Files)
start_dask = time.time()
history_manager_dask = dd.read_csv('./data/manager/clean_Managers_part*.csv')
history_manager_dask = history_manager_dask.compute()  # Trigger actual computation
end_dask = time.time()
print("Dask Load Time:", end_dask - start_dask)

# 4. Modin (Parallelized Pandas for Distributed Computing)
start_modin = time.time()
history_manager_modin = mpd.read_csv('./data/manager/clean_Managers_part*.csv')
end_modin = time.time()
print("Modin Load Time:", end_modin - start_modin)

# 5. PyArrow (For Parquet/ORC Formats)
start_pyarrow = time.time()
files = glob.glob('./data/manager/clean_Managers_part*.csv')
tables = [pv_csv.read_csv(file) for file in files]
combined_table = pq.concat_tables(tables)  # Combine the tables read from CSVs
pq.write_table(combined_table, './data/manager/clean_Managers.parquet')
end_pyarrow = time.time()
print("PyArrow CSV to Parquet Conversion Time:", end_pyarrow - start_pyarrow)

# Later for loading the Parquet file
start_parquet = time.time()
df_parquet = pq.read_table('./data/manager/clean_Managers.parquet').to_pandas()
end_parquet = time.time()
print("PyArrow Parquet Load Time:", end_parquet - start_parquet)

# 6. SQLite (For Storing Data and Querying Later)
conn = sqlite3.connect('data.db')
start_sqlite = time.time()
# Use Dask or Pandas for large file batch insertions into SQLite
df = pd.read_csv('./data/manager/clean_Managers_part*.csv')
df.to_sql('managers', conn, if_exists='replace', index=False, chunksize=10000)  # insert in chunks
end_sqlite = time.time()
print("SQLite Load Time:", end_sqlite - start_sqlite)
