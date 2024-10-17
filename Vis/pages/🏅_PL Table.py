import streamlit as st
import pandas as pd
import sys
import os
import altair as alt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'FPL')))
from fpl_api_collection import (
    get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data
)
from fpl_utils import (
    define_sidebar
)
st.set_page_config(page_title='PL Table', page_icon=':sports-medal:', layout='wide')
define_sidebar()
st.title('Premier League Table')

league_df = get_league_table()

team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()

ct_gw = get_current_gw()

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]

league_df = league_df.join(new_fixt_df)

float_cols = league_df.select_dtypes(include='float64').columns.values

league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1

league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

## Very slow to load, works but needs to be sped up.
def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict
    
    merged_dict = {}
    merged_dict[1.5] = []
    merged_dict[2.5] = []
    merged_dict[3.5] = []
    merged_dict[4.5] = []
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        decoupled_list = list(set(v))
        merged_dict[k] = decoupled_list
    for i in range(1,6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict


home_away_dict = get_home_away_str_dict()

def color_fixtures(val):
    bg_color = 'background-color: '
    font_color = 'color: '
    if val in home_away_dict[1]:
        bg_color += '#147d1b'
    if val in home_away_dict[1.5]:
        bg_color += '#0ABE4A'
    elif val in home_away_dict[2]:
        bg_color += '#00ff78'
    elif val in home_away_dict[2.5]:
        bg_color += "#caf4bd"
    elif val in home_away_dict[3]:
        bg_color += '#eceae6'
    elif val in home_away_dict[3.5]:
        bg_color += "#fa8072"
    elif val in home_away_dict[4]:
        bg_color += '#ff0057'
        font_color += 'white'
    elif val in home_away_dict[4.5]:
        bg_color += '#C9054F'
        font_color += 'white'
    elif val in home_away_dict[5]:
        bg_color += '#920947'
        font_color += 'white'
    else:
        bg_color += ''
    style = bg_color + '; ' + font_color
    return style

for col in new_fixt_cols:
    if league_df[col].dtype == 'O':
        max_length = league_df[col].str.len().max()
        if max_length > 7:
            league_df.loc[league_df[col].str.len() <= 7, col] = league_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')

st.dataframe(league_df.style.applymap(color_fixtures, subset=new_fixt_cols) \
             .format(subset=float_cols, formatter='{:.2f}'), height=740, width=None)

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
if model_option == "Overall":
    model_type = ""
elif model_option == "Home":
    model_type = "home"
else:  # "Away"
    model_type = "away"

# Display DataFrame
rating_df = teams_df.sort_values("ovr_rating" + ("_" + model_type if model_type else ""), ascending=False)

# Get the maximum values for each rating without converting to percentage
max_ovr = rating_df["ovr_rating" + ("_" + model_type if model_type else "")].max()
max_o = rating_df["o_rating" + ("_" + model_type if model_type else "")].max()
max_d = rating_df["d_rating" + ("_" + model_type if model_type else "")].max()

# Set up columns for layout
df_col, chart_col = st.columns([24, 24])  # Adjust the column sizes as needed

# Configure progress columns for ratings with actual values
column_config = {
    "ovr_rating" + ("_" + model_type if model_type else ""): st.column_config.ProgressColumn(label="Overall Rating", max_value=max_ovr),
    "o_rating" + ("_" + model_type if model_type else ""): st.column_config.ProgressColumn(label="Offensive Rating", max_value=max_o),
    "d_rating" + ("_" + model_type if model_type else ""): st.column_config.ProgressColumn(label="Defensive Rating", max_value=max_d),
}

with df_col:
    # Display the DataFrame with full width using progress columns
    st.dataframe(
        rating_df[["name", 
                    "ovr_rating" + ("_" + model_type if model_type else ""),
                    "o_rating" + ("_" + model_type if model_type else ""),
                    "d_rating" + ("_" + model_type if model_type else "")]],
        hide_index=True,
        use_container_width=True,  # This makes the DataFrame take full width
        column_config=column_config  # Apply the progress column configuration
    )

    
# Scatter plot setup
x_domain = [teams_df["d_rating" + ("_" + model_type if model_type else "")].min()-0.1, teams_df["d_rating" + ("_" + model_type if model_type else "")].max() + 0.1]
y_range = [teams_df["o_rating" + ("_" + model_type if model_type else "")].min()-100, teams_df["o_rating" + ("_" + model_type if model_type else "")].max() + 100]

# Create scatter plot with reduced size
scatter_plot = (
    alt.Chart(teams_df, height=400, width=500)  # Adjust height and width here
    .mark_point(filled=True, size=100)
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

# Combine all chart elements
with chart_col:
    st.altair_chart(scatter_plot + off_mean_line + def_mean_line, use_container_width=True)
