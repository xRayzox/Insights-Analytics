import streamlit as st
from fpl_utils.fpl_api_collection import get_total_fpl_players
import requests

total_players = get_total_fpl_players()

def define_sidebar():
    st.sidebar.subheader('About')
    st.sidebar.write("""This website is designed to help you analyse and
                     ultimately pick the best Fantasy Premier League Football
                     options for your team.""")
    st.sidebar.write("""Current number of FPL Teams: """ + str('{:,.0f}'.format(total_players)))
    st.sidebar.write('[Author](https://www.linkedin.com/in/tim-youell-616731a6)')
    st.sidebar.write('[GitHub](https://github.com/TimYouell15)')


def get_annot_size(sl1, sl2):
    ft_size = sl2 - sl1
    if ft_size >= 24:
        annot_size = 2
    elif (ft_size < 24) & (ft_size >= 16):
        annot_size = 3
    elif (ft_size < 16) & (ft_size >= 12):
        annot_size = 4
    elif (ft_size < 12) & (ft_size >= 9):
        annot_size = 5
    elif (ft_size < 9) & (ft_size >= 7):
        annot_size = 6
    elif (ft_size < 7) & (ft_size >= 5):
        annot_size = 7
    else:
        annot_size = 8
    return annot_size


def get_rotation(sl1, sl2):
    diff = sl2 - sl1
    if diff < 7:
        rotation = 0
    else:
        rotation = 90
    return rotation


def map_float_to_color(val, cmap, min_value, max_value):
    """
    Map a float value to a hashed color from a custom colormap represented as a list of hashed colors within a specific range.

    Args:
        value (float): The float value to map to a color (between min_value and max_value).
        cmap (list): A custom list of hashed colors to use as the colormap.
        min_value (float): The minimum value in the range.
        max_value (float): The maximum value in the range.

    Returns:
        str: The hashed color corresponding to the input float value.
    """
    value = max(min_value, min(max_value, val))
    normalized_value = (value - min_value) / (max_value - min_value)
    index = min(int(normalized_value * (len(cmap))), len(cmap) - 1)
    return cmap[index]


def chip_converter(name):
    if name == '3xc':
        return 'Triple Captain'
    if name == 'bboost':
        return 'Bench Boost'
    if name == 'freehit':
        return 'Free Hit'
    if name == 'wildcard':
        return 'Wildcard'


def get_text_color_from_hash(hash_color):
    color_map = {
        '#920947': 'white',
        '#ff0057': 'white',
        '#fa8072': 'white',
        '#147d1b': 'white'
    }
    return color_map.get(hash_color, 'black')


def get_user_timezone():
    try:
        ip = requests.get('https://api.ipify.org').text
        response = requests.get(f'https://ipinfo.io/{ip}/json')
        data = response.json()
        return data['timezone']
    except Exception as e:
        return 'Africa/Tunis'