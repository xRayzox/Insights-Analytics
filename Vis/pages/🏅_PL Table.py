import streamlit as st
import pandas as pd
import sys
import os
import tempfile
from io import BytesIO
import altair as alt
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from plottable import ColumnDefinition, Table
from plottable.plots import image
import urllib.request
from PIL import Image
import base64
from io import BytesIO


pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'FPL')))
from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)
from fpl_utils import (
    define_sidebar
)

with open('./data/wave.css') as f:
        css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
########################################################
st.set_page_config(page_title='PL Table', page_icon=':sports-medal:', layout='wide')
define_sidebar()
st.title('Premier League Table')

@st.cache_data(persist="disk")
def load_image_from_url(url):
    """Load an image from a URL and save it to a temporary file."""
    # Create a temporary filename
    temp_filename = f"temp_{os.path.basename(url)}"
    
    try:
        # Load image data directly into memory
        with urllib.request.urlopen(url) as response:
            image_data = response.read()  # Read image data into memory
        
        # Open the image from the BytesIO stream
        image = Image.open(BytesIO(image_data)).convert("RGBA")
        
        # Save the image to a temporary file
        image.save(temp_filename)
        
        return temp_filename
    except Exception as e:
        print(f"Error loading image from URL {url}: {e}")
        return None


def form_color(form):
    """Map characters in a form string to colors."""
    color_mapping = {
        'W': '#28a745',  # Green for Win
        'D': '#ffc107',  # Orange for Draw
        'L': '#dc3545',  # Red for Loss
    }
    
    # Create a list of colors for the form string
    return [color_mapping[char] for char in form if char in color_mapping]

def custom_plot_fn_form(ax: plt.Axes, val):
    """Plot the form string on a Matplotlib Axes object with color coding."""
    colors = form_color(val)  # Get the list of colors for the form
    num_chars = len(val)  # Number of characters in the form
    spacing = 0.2  # Adjust this value for more/less horizontal spacing

    # Calculate the horizontal positions based on the number of characters
    for i, (char, color) in enumerate(zip(val, colors)):
        x_pos = 0.5 + (i - (num_chars - 1) / 2) * spacing  # Centering the characters
        
        # Draw the character with its corresponding color
        ax.text(x_pos, 0.5, char, fontsize=14, ha='center', va='center',
                bbox=dict(facecolor=color, alpha=0.5))
        
@st.cache_data(persist="disk")
def download_image_to_temp(url):
    """Download an image from a URL and convert it to a base64 string."""
    if url:  # Ensure the URL is not None
        try:
            with urllib.request.urlopen(url) as response:
                image = Image.open(response).convert("RGBA")
                # Use NamedTemporaryFile to save the image temporarily
                with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp_file:
                    image.save(tmp_file, format='PNG')  # Save image in PNG format
                    tmp_file.seek(0)  # Move to the beginning of the file
                    # Convert the image to base64 for displaying in the Altair chart
                    return "data:image/png;base64," + base64.b64encode(tmp_file.read()).decode()
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            return None
    return None

# --- Data Loading and Processing ---
@st.cache_data
def fetch_data():
    league_df = get_league_table()
    team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
    ct_gw = get_current_gw()
    return league_df,team_fdr_df, team_fixt_df, team_ga_df, team_gf_df,ct_gw
league_df, team_fdr_df, team_fixt_df, team_ga_df, team_gf_df,ct_gw = fetch_data()


new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols
league_df = league_df.join(new_fixt_df)
float_cols = league_df.select_dtypes(include='float64').columns.values
league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1
league_df['GD'] = league_df['GD'].map('{:+}'.format)
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + "@x2.png"
teams_df['logo_image'] = teams_df['logo_url'].apply(load_image_from_url)
team_logo_mapping = pd.Series(teams_df['logo_image'].values, index=teams_df['short_name']).to_dict()
# Map each team's logo image to the league DataFrame
league_df['logo_team'] = league_df['Team'].map(team_logo_mapping)
# Calculate and assign rankings in the league DataFramae
league_df['Rank'] = league_df['Pts'].rank(ascending=False, method='min').astype(int)
if 'logo_base64' not in teams_df.columns:
    teams_df['logo_base64'] = teams_df['logo_url'].apply(download_image_to_temp)

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length + 9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict

    merged_dict = {k: [] for k in [1.5, 2.5, 3.5, 4.5]}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        merged_dict[k] = list(set(v))
    for i in range(1, 6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict
home_away_dict = get_home_away_str_dict()
def color_fixtures(val):
    color_map = {
        1: "#257d5a",
        2: "#00ff86",
        3: "#ebebe4",
        4: "#ff005a",
        5: "#861d46",
    }
    for key in color_map:
        if val in home_away_dict[key]:
            return color_map[key]
    return "#000000"  # Default color if no match


# Assuming league_df is defined and populated.


# Modify cmap for Fixture Column Definitions
def fixture_cmap(val):
    return color_fixtures(val)  # Directly return the color

def custom_plot_fn(ax: plt.Axes, val):
    ax.text(0.5, 0.5, str(val), fontsize=14, ha='center', va='center',
        bbox=dict(facecolor=color_fixtures(val), alpha=0.5))

# --- Table Styling ---
bg_color = "#FFFFFF"
text_color = "#000000"

row_colors = {
    "top4": "#E1FABC",
    "top6": "#FFFC97",
    "relegation": "#E79A9A",
    "even": "#E2E2E1",
    "odd": "#B3B0B0"
}


matplotlib.rcParams["text.color"] = text_color
matplotlib.rcParams["font.family"] = "monospace"

# --- Column Definitions ---
col_defs = [
    ColumnDefinition(
        name="Rank",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="logo_team",
        textprops={'ha': "center", 'va': "center", 'color': "white"},
        plot_fn=image,
        width=1,
    ),
    ColumnDefinition(
        name="Team",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="GP",
        group="Matches Played",

        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="W",
        group="Matches Played",

        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="D",
        group="Matches Played",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="L",
        group="Matches Played",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GF",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GA",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="GD",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="CS",
        group="Goals",
        textprops={'ha': "center"},
        width=0.5
    ),
    ColumnDefinition(
        name="Pts",
        group="Points",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="Pts/Game",
        group="Points",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="Form",
        group="Points",
        textprops={'ha': "center"},
        plot_fn=custom_plot_fn_form,
        width=2
    ),
    ColumnDefinition(
        name="GF/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="GA/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    ColumnDefinition(
        name="CS/Game",
        group="ByGame",
        textprops={'ha': "center"},
        width=1
    ),
    
]

# Modify Fixture Column Definitions
for gw in range(ct_gw, ct_gw + 3):
    col_defs.append(
        ColumnDefinition(
            name=f"GW{gw}",
            group="Fixtures",
            textprops={'ha': "center"},
            width=1,
            plot_fn=custom_plot_fn  # Use the custom plotting function
        )
    )


# --- Plottable Table ---
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust figsize for Streamlit
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)



table = Table(
    league_df,
    column_definitions=col_defs,
    columns=['logo_team','Team', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'CS', 'Pts', 
             'Pts/Game','Form', 'GF/Game', 'GA/Game', 'CS/Game', f'GW{ct_gw}', f'GW{ct_gw+1}', f'GW{ct_gw+2}'], 
    index_col="Rank",
    row_dividers=True,
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    footer_divider=True,
    textprops={"fontsize": 14},  # Adjust fontsize for Streamlit
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": .5, "linestyle": "-"},
    ax=ax
)


for idx in range(len(league_df)):
    if league_df.iloc[idx]['Rank'] <= 4:
        table.rows[idx].set_facecolor(row_colors["top4"])
    elif league_df.iloc[idx]['Rank'] <= 6:
        table.rows[idx].set_facecolor(row_colors["top6"])
    elif league_df.iloc[idx]['Rank'] >= 18:  # Assuming relegation zone starts at 18
        table.rows[idx].set_facecolor(row_colors["relegation"])



# --- Display the Table in Streamlit ---
st.pyplot(fig)
####################################################


# Set the title and caption
st.title("Team Offensive / Defensive Ratings")
st.caption("Compare overall, offensive, and defensive strengths of teams.")

# Calculate ratings using the specified strength columns
teams_df["ovr_rating_home"] = teams_df["strength_overall_home"]
teams_df["ovr_rating_away"] = teams_df["strength_overall_away"]
teams_df["o_rating_home"] = teams_df["strength_attack_home"]
teams_df["o_rating_away"] = teams_df["strength_attack_away"]
teams_df["d_rating_home"] = teams_df["strength_defence_home"]
teams_df["d_rating_away"] = teams_df["strength_defence_away"]

# Overall ratings calculation
teams_df["ovr_rating"] = (teams_df["ovr_rating_home"] + teams_df["ovr_rating_away"]) / 2
teams_df["o_rating"] = (teams_df["o_rating_home"] + teams_df["o_rating_away"]) / 2
teams_df["d_rating"] = (teams_df["d_rating_home"] + teams_df["d_rating_away"]) / 2

# Options for ratings
model_option = st.selectbox("Data Source", ("Overall", "Home", "Away"))
model_type = "home" if model_option == "Home" else "away" if model_option == "Away" else ""

# Display DataFrame
rating_df = teams_df.sort_values("ovr_rating" + ("_" + model_type if model_type else ""), ascending=False)

# Convert rating columns to float
rating_df["ovr_rating" + ("_" + model_type if model_type else "")] = rating_df["ovr_rating" + ("_" + model_type if model_type else "")].astype(float)
rating_df["o_rating" + ("_" + model_type if model_type else "")] = rating_df["o_rating" + ("_" + model_type if model_type else "")].astype(float)
rating_df["d_rating" + ("_" + model_type if model_type else "")] = rating_df["d_rating" + ("_" + model_type if model_type else "")].astype(float)

# Get the maximum values for each rating type
max_ovr = rating_df["ovr_rating" + ("_" + model_type if model_type else "")].max()
max_o = rating_df["o_rating" + ("_" + model_type if model_type else "")].max()
max_d = rating_df["d_rating" + ("_" + model_type if model_type else "")].max()

# Get min and max for d_rating and o_rating
d_rating_min = teams_df["d_rating" + ("_" + model_type if model_type else "")].min()
d_rating_max = teams_df["d_rating" + ("_" + model_type if model_type else "")].max()
o_rating_min = teams_df["o_rating" + ("_" + model_type if model_type else "")].min()
o_rating_max = teams_df["o_rating" + ("_" + model_type if model_type else "")].max()

# Calculate ranges with increased margins
x_margin = (d_rating_max - d_rating_min) * 0.05  # 5% of the range
y_margin = (o_rating_max - o_rating_min) * 0.1    # 10% of the range

# Define the x_domain and y_range with calculated margins
x_domain = [d_rating_min - x_margin, d_rating_max + x_margin]
y_range = [o_rating_min - y_margin, o_rating_max + y_margin]

# Create scatter plot
scatter_plot = (
    alt.Chart(teams_df, height=400, width=800)
    .mark_image(width=30, height=30)  # Adjust size as needed
    .encode(
        x=alt.X(
            "d_rating" + ("_" + model_type if model_type else ""),
            type="quantitative",
            title="Defensive Rating",
            scale=alt.Scale(domain=x_domain),
        ),
        y=alt.Y(
            "o_rating" + ("_" + model_type if model_type else ""),
            type="quantitative",
            title="Offensive Rating",
            scale=alt.Scale(domain=y_range),
        ),
        tooltip=[
            alt.Tooltip("name", title="Team"),
            alt.Tooltip("ovr_rating" + ("_" + model_type if model_type else ""), title="Overall Rating", format="d"),
            alt.Tooltip("o_rating" + ("_" + model_type if model_type else ""), title="Offensive Rating", format="d"),
            alt.Tooltip("d_rating" + ("_" + model_type if model_type else ""), title="Defensive Rating", format=".2f"),
        ],
        url='logo_base64',
    )
)

# Mean lines
off_mean_line = (
    alt.Chart(pd.DataFrame({"Mean Offensive Rating": [teams_df["o_rating" + ("_" + model_type if model_type else "")].mean()]}))
    .mark_rule(color="#60b4ff", opacity=0.66)
    .encode(y="Mean Offensive Rating")
)

def_mean_line = (
    alt.Chart(pd.DataFrame({"Mean Defensive Rating": [teams_df["d_rating" + ("_" + model_type if model_type else "")].mean()]}))
    .mark_rule(color="#60b4ff", opacity=0.66)
    .encode(x="Mean Defensive Rating")
)

# Display the chart
st.altair_chart(scatter_plot + off_mean_line + def_mean_line, use_container_width=True)

