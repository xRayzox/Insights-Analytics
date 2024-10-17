import pytz

MY_FPL_ID = 3764749
BASE_URL = 'https://fantasy.premierleague.com/api/'
AUTHOR_CONTINENT = 'Africa'
AUTHOR_CITY = 'Tunis'
TZS = pytz.all_timezones
CONTINENT_LIST = ['Africa', 'America', 'Asia', 'Australia', 'Brazil',
                  'Canada', 'Europe', 'Indian', 'Pacific']
TIMEZONES_BY_CONTINENT = {
    'Africa': [tz for tz in TZS if tz.startswith('Africa')],
    'America': [tz for tz in TZS if tz.startswith('America')],
    'Asia': [tz for tz in TZS if tz.startswith('Asia')],
    'Australia': [tz for tz in TZS if tz.startswith('Australia')],
    'Brazil': [tz for tz in TZS if tz.startswith('Brazil')],
    'Canada': [tz for tz in TZS if tz.startswith('Canada')],
    'Europe': [tz for tz in TZS if tz.startswith('Europe')],
    'Indian': [tz for tz in TZS if tz.startswith('Indian')],
    'Pacific': [tz for tz in TZS if tz.startswith('Pacific')],
    'Other': [tz for tz in TZS if not any(tz.startswith(cont) for cont in CONTINENT_LIST)]
}
