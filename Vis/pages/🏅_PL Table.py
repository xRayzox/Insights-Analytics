import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'FPL')))
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

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw + 2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw + 2)]

league_df = league_df.join(new_fixt_df)

float_cols = league_df.select_dtypes(include='float64').columns.values

league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1

league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df['logo_url'] = "https://resources.premierleague.com/premierleague/badges/70/t" + teams_df['code'].astype(str) + ".png"
team_logo_mapping = pd.Series(teams_df.logo_url.values, index=teams_df.short_name).to_dict()

# Create a new column for logos and team names
league_df['Team'] = league_df['Team'].apply(lambda x: f'<img src="{team_logo_mapping[x]}" width="20"/> {x}')

## Slow to load part
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
    for i in range(1, 6):
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
            league_df.loc[league_df[col].str.len() <= 7, col] = league_df[col].str.pad(width=max_length + 9, side='both', fillchar=' ')

# Render the league table with logos in the Team column
st.dataframe(league_df.style.applymap(color_fixtures, subset=new_fixt_cols) \
             .format(subset=float_cols, formatter='{:.2f}').set_table_attributes('style="width: 100%; table-layout: auto;"'), height=740, width=None)
