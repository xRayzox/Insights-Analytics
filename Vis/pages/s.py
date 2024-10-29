import streamlit as st
from bokeh.models import ColumnDataSource, TableColumn, DataTable

# --- Streamlit Configuration ---
st.set_page_config(layout="wide")

# --- Sample Data (replace with your actual data) ---
data = {
    'Rank': [1, 2, 3, 4, 5],
    'Team': ['Manchester City', 'Arsenal', 'Manchester United', 'Newcastle', 'Liverpool'],
    'GP': [38, 38, 38, 38, 38],
    'Pts': [89, 84, 75, 71, 67]
    # Add other columns as needed 
}

# Create a Pandas DataFrame (optional but recommended)
df = pd.DataFrame(data)

# --- Bokeh Table ---
source = ColumnDataSource(df)  # Pass the DataFrame directly

columns = [
    TableColumn(field='Rank', title='Rank'),
    TableColumn(field='Team', title='Team'),
    TableColumn(field='GP', title='GP'),
    TableColumn(field='Pts', title='Pts'),
    # Add more TableColumn objects for other columns
]

data_table = DataTable(source=source, columns=columns, width=800, height=400)

# --- Streamlit App ---
st.title("Simple Premier League Table")
st.bokeh_chart(data_table)