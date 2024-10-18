#importing SBopen class from mplsoccer to open the data
from mplsoccer import Sbopen
# The first thing we have to do is open the data. We use a parser SBopen available in mplsoccer.
parser = Sbopen()

#opening data using competition method
df_competition = parser.competition()
#structure of data
print(df_competition.info())
