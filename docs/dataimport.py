import pybaseball as pyb
import pandas as pd
import numpy as np
import math
import collections
import matplotlib
import matplotlib.pyplot as pp
import geopy as gp
from geopy import geocoders

# fgteams: a dict with how FanGraphs filtering maps teams, can eventually be converted into a function that automatically maps these based on year.
# Will essentially be the same order from 1998 on, but different values for teams
fgteams = {1: "LAA", 2: "BAL", 3: "BOS", 4: "CHW", 5: "CLE", 6: "DET", 7: "KCR",
           8: "MIN", 9: "NYY", 10: "OAK", 11: "SEA", 12: "TBR", 13: "TEX", 14: "TOR",
           15: "ARI", 16: "ATL", 17: "CHC", 18: "CIN", 19: "COL", 20: "MIA", 21: "HOU",
           22: "LAD", 23: "MIL", 24: "WSN", 25: "NYM", 26: "PHI", 27: "PIT", 28: "STL",
           29: "SDP", 30: "SFG"}

# hitting_cols: the columns to be sourced from FanGraphs (via. batting_stats()) for hitting data (player or team)
# does not include IDfg, Name, Age, G
hitting_cols = ['AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI',
        'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'AVG', 'GB', 'FB', 'LD', 'IFFB',
        'IFH', 'BU', 'BUH', 'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'GB/FB', 'LD%',
        'GB%', 'FB%', 'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'wOBA', 'wRAA', 'wRC', 'WAR', 'Spd', 'wRC+',
        'WPA', '-WPA', '+WPA', 'pLI', 'phLI', 'PH', 'WPA/LI', 'Clutch', 'O-Swing%', 'Z-Swing%', 
        'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'SwStr%', 'Pace', 'wSB', 'UBR',
        'Off', 'wGDP','Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'TTO%', 'EV', 'LA', 'Barrels',
        'Barrel%', 'maxEV', 'HardHit', 'HardHit%', 'CStr%', 'xBA', 'xSLG', 'xwOBA']

# pitching_cols: the columns to be sourced from FanGraphs (via. pitching_stats()) for pitching data (player or team)
pitching_cols = ['W' , 'L', 'ERA', 'CG', 'ShO', 'SV', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR',
                 'BB', 'IBB', 'BK', 'SO', 'GB', 'FB', 'LD', 'IFFB', 'Balls', 'Strikes', 'Pitches', 'RS', 
                 'IFH', 'BU', 'BUH', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%',
                 'FIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'WAR', 'tERA', 'xFIP',
                 'WPA', '-WPA', '+WPA', 'pLI', 'inLI', 'exLI', 'Clutch', 'FBv', 'SLv', 'CTv', 'CBv', 'CHv', 'SFv',
                 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
                 'SwStr%', 'SD', 'MD', 'HLD', 'ERA-', 'FIP-', 'xFIP-', 'K%', 'BB%', 'SIERA', 'RS/9', 'E-F', 'K-BB%',
                 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'kwERA', 'FRM', 'Barrel%', 'HardHit%', 'CStr%',
                 'CSW%', 'xERA']

# fielding_cols_gen: fielding columns to be added to all fielding tables
fielding_cols_gen = ['Pos','PO','A','E','FE','TE','DP','DPS','DPT','DPF','FP','DRS','Def']

# other fielding_cols: to be used for filling in position-specific columns of FanGraphs fielding stats
fielding_cols_C = ['SB','CS','PB','WP','rSB']
fielding_cols_IF = []

# drop_bio: columns to exclude from player biographical information
drop_bio = ['deathYear','deathMonth','deathDay','deathCountry','deathState','deathCity','nameFirst','nameLast','nameGiven',
            'debut','finalGame','retroID','bbrefID']

month = {
    'Mar':'03',
    'Apr':'04',
    'May':'05',
    'Jun':'06',
    'Jul':'07',
    'Aug':'08',
    'Sep':'09',
    'Oct':'10'
    }

'''
genID(df)
    INPUT
df: FanGraphs DataFrame (batting or pitching)
    OUTPUT
DataFrame of ID's of players in df
'''
def genID(df):
    return df.reset_index()[['key_mlbam', 'Name', 'Team']].drop_duplicates(subset=['key_mlbam']).merge(pyb.chadwick_register()
                                                                                                .rename(columns={'playerID':'key_bbref', 'mlb_played_first':'debut', 'mlb_played_last':'recent_season'})
                                                                                                .drop(columns=['name_last', 'name_first']), on='key_mlbam', how='left')
    
'''
genBio(IDdf)
    INPUT
IDdf: ID DataFrame
    OUTPUT
DataFrame with biographical information about players
'''
def genBio(IDdf):
    return IDdf.merge(pyb.lahman.people().reset_index()
                                         .rename(columns={'playerID':'key_bbref'})
                                         .drop(columns=drop_bio), on='key_bbref', how='left').drop(columns=['index','key_fangraphs', 'key_retro', 'key_bbref'])

'''
gameLogs(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
DataFrame with gamelogs for a given timeframe
'''
def gameLogs(start_year, end_year):
    years = []
    repot = pyb.lahman.teams_core().groupby('yearID')
    for year in range(start_year,end_year+1):
        temp = pd.concat([pyb.schedule_and_record(year,team) for team in repot.get_group(year)['teamIDBR']])
        temp['Season'] = year
        years.append(temp)
    gl = pd.concat(years)
    gl['R_tot'] = gl['R'] + gl['RA']
    gl['R_diff'] = gl.apply(lambda x: x['R']-x['RA'] if x['R']>x['RA'] else x['RA']-x['R'],axis=1)
    gl['Date_split'] = gl['Date'].apply(lambda x: x.split(',')[1])
    gl['Day'] = gl['Date_split'].apply(lambda x: x.split(' ')[2])
    gl['Day'] = gl['Day'].apply(lambda x: '0'+x if int(x)<10 else x)
    gl['Month'] = gl['Date_split'].apply(lambda x: x.split(' ')[1])
    gl['Month'] = gl['Month'].map(dict)
    gl['Time'] = gl['Time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    return gl[['Day','Month','Season','Tm','Opp','Time','D/N','Attendance']]

'''
teamDepot(start_year, end_year)
'''
def teamDepot(start_year, end_year):
    div={
        'E':'EAST',
        'C':'CENTRAL',
        'W':'WEST'
    }
    lmn = pyb.lahman.teams_core().groupby('yearID')
    teams = pd.concat([lmn.get_group(year) for year in range(start_year,end_year+1)])
    teams['divID'] = teams['divID'].map(div)
    return teams[['teamIDBR','name','yearID','divID','Rank','W','L','BPF','PPF']].rename(columns={'teamIDBR':'Team',
                                                                                                  'name':'TeamName',
                                                                                                  'yearID':'Season',
                                                                                                  'divID':'Division'})

'''
fgBatting(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
fgdf: DF with FanGraphs data for given timeframe (index: key_mlbam)
'''
def fgBatting(start_year, end_year):
    team_dfs = []

    for team in fgteams:
        fg_temp = pyb.batting_stats(start_year, 
                                           end_year, 
                                           team=team, 
                                           qual=10, 
                                           split_seasons = True)[['IDfg', 'Name', 'Age', 'G', 'Season'] + 
                                                                  [col for col in hitting_cols]]
        fg_temp['Team'] = fgteams[team]
        team_dfs.append(fg_temp)
    fgdf = pd.concat(team_dfs).rename(columns={"IDfg":"key_fangraphs"}).merge(pyb.chadwick_register()[['key_fangraphs', 'key_mlbam']])
    fgdf = fgdf[['key_mlbam','Name', 'Age', 'Team','G', 'Season'] + [col for col in hitting_cols]].set_index('key_mlbam')
    return fgdf

'''
statBatting(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
statdf: DataFrame with Statcast data for given timeframe
'''
def statBatting(start_year, end_year):
    years = []
    for year in range(start_year,end_year+1):
        stat_temp = pyb.statcast_batter_exitvelo_barrels(year=year, minBBE=1).rename(columns={"player_id":"key_mlbam"}).drop(columns=["last_name","first_name"]).merge(pyb.statcast_batter_expected_stats(year=year, minPA=10)
                                                                                                                                                                                                                            .rename(columns={"player_id":"key_mlbam"})
                                                                                                                                                                                                                            .drop(columns=["last_name","first_name"]), on="key_mlbam", how="left")
        stat_temp['Season'] = year
        years.append(stat_temp)
    statdf = pd.concat(years)
    return statdf

'''
bwarBatting(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
DataFrame with bWAR data (from BaseballReference) for a given timeframe
'''
def bwarBatting(start_year, end_year):
    bwar = pyb.bwar_bat(return_all=True).rename(columns={"mlb_ID":"key_mlbam"}).groupby(['year_ID'])
    return pd.concat([bwar.get_group(year) for year in range(start_year,end_year+1)]).drop(columns=['name_common', 'player_ID', 'age', 'lg_ID']).rename(columns={'year_ID':'Season',
                                                                                                                                                                 'team_ID':'Team',
                                                                                                                                                                 'lg_ID':'League'})

'''
teamBatting(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
DataFrame with FanGraphs team batting data
'''
def teamBatting(start_year, end_year):
    return pd.concat([pyb.team_batting(year)[['teamIDfg', 'Team', 'Season'] + [col for col in hitting_cols]].merge(pyb.lahman.teams_upstream()[['teamIDBR', 'BPF', 'yearID']].rename(columns={'teamIDBR':'Team','yearID':'Season'}),on=['Team','Season'], how='left').merge(pyb.team_pitching(year, split_seasons=True)[['Season', 'Team', 'W', 'L']],  on=['Team','Season']) for year in range(start_year, end_year+1)]).reset_index()

'''
fgPitching(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
fgdf: DF with FanGraphs data for given timeframe (index: key_mlbam)
'''
def fgPitching(start_year, end_year):
    team_dfs = []

    for team in fgteams:
        fg_temp = pyb.pitching_stats(start_year, 
                                           end_year, 
                                           team=team, 
                                           qual=10, 
                                           split_seasons = True)[['IDfg', 'Name', 'Age', 'Season', 'G', 'GS'] + 
                                                                  [col for col in pitching_cols]]
        fg_temp['Team'] = fgteams[team]
        team_dfs.append(fg_temp)
    fgdf = pd.concat(team_dfs).rename(columns={"IDfg":"key_fangraphs"}).merge(pyb.chadwick_register()[['key_fangraphs', 'key_mlbam']])
    fgdf = fgdf[['key_mlbam','Name', 'Age', 'Team', 'Season', 'G', 'GS'] + [col for col in pitching_cols]].set_index('key_mlbam')
    
    return fgdf

'''
statPitching(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
statdf: DataFrame with Statcast data for given timeframe
'''
def statPitching(start_year, end_year):
    return pd.concat([pyb.statcast_pitcher_exitvelo_barrels(year=year,minBBE=50)
                         .merge(pyb.statcast_pitcher_expected_stats(year=year, minPA=50)
                                   .drop(columns=['last_name','first_name']),on='player_id',how='left')
                         .merge(pyb.statcast_pitcher_pitch_arsenal(year=2022,minP=200,arsenal_type='avg_speed')
                                   .drop(columns=['last_name','first_name'])
                                   .rename(columns={'pitcher':'player_id'}),on='player_id',how='left')
                         .merge(pyb.statcast_pitcher_pitch_arsenal(year=year,minP=200,arsenal_type='avg_spin')
                                   .drop(columns=['last_name','first_name'])
                                   .rename(columns={'pitcher':'player_id'}),on='player_id',how='left')
                         .drop(columns=['last_name','first_name'])
                         .rename(columns={'player_id':'key_mlbam',
                                          'year':'Season'}) for year in range(start_year,end_year+1)])

'''
bWARPitching(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
DataFrame with bWAR data (from BaseballReference) for a given timeframe
'''
def bwarPitching(start_year, end_year):
    bwar = pyb.bwar_pitch(return_all=True).rename(columns={"mlb_ID":"key_mlbam"}).groupby(['year_ID'])
    return pd.concat([bwar.get_group(year) for year in range(start_year,end_year+1)]).drop(columns=['name_common', 'player_ID', 'age', 'lg_ID']).rename(columns={'year_ID':'Season',
                                                                                                                                                                 'team_ID':'Team',
                                                                                                                                                                 'lg_ID':'League'})
'''
teamPitching(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
DataFrame with FanGraphs team batting data
'''
def teamPitching(start_year, end_year):
    return pyb.team_pitching(start_year, end_year, ind=1)

'''
genBatting(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
fg: DataFrame of Fangraphs batting data from a given timeframe
stat: DataFrame of Statcast batting data from a given timeframe
bwar: DataFrame of Baseball Reference bWAR data from a given timeframe
team: DataFrame of team Fangraphs batting data from a given timeframe
bio: DataFrame of biographical data from a given timeframe
df_id: DataFrame of player ID's from a given timeframe
'''
def genBatting(start_year, end_year):
    fg = fgBatting(start_year,end_year)
    df_id = genID(fg)
    bio = genBio(df_id)
    stat = fg.reset_index()[['key_mlbam','Season']].merge(statBatting(start_year, end_year), on=['key_mlbam', 'Season'], how='left').drop_duplicates()
    bwar = fg.reset_index()[['key_mlbam','Season','Team']].merge(bwarBatting(start_year, end_year), on=['key_mlbam','Season','Team'],how='left')
    team = teamBatting(start_year, end_year)
    return [fg, stat, bwar, team, bio, df_id]

'''
genPitching(start_year, end_year)
    INPUT
start_year: beginning of year range to pull data
end_year: end of year range to pull data
    OUTPUT
fg: DataFrame of Fangraphs pitching data from a given timeframe
stat: DataFrame of Statcast pitching data from a given timeframe
bwar: DataFrame of Baseball Reference bWAR data from a given timeframe
team: DataFrame of team Fangraphs pitching data from a given timeframe
bio: DataFrame of biographical data from a given timeframe
df_id: DataFrame of player ID's from a given timeframe
'''
def genPitching(start_year,end_year):
    fg = fgPitching(start_year,end_year)
    df_id = genID(fg)
    bio = genBio(df_id)
    stat = fg.reset_index()[['key_mlbam','Season']].merge(statPitching(start_year,end_year), on=['key_mlbam','Season'], how='left').drop_duplicates()
    bwar = fg.reset_index()[['key_mlbam','Season','Team']].merge(bwarPitching(start_year, end_year), on=['key_mlbam','Season','Team'],how='left')
    team = teamPitching(start_year, end_year)
    return [fg, stat, bwar, team, bio, df_id]

'''
# genFielding: generates fielding data
def genFielding(bio, year):
    return pyb.fielding_stats(year, qual=10).rename(columns={"IDfg":"key_fangraphs"}).set_index("key_fangraphs").drop(columns={"Name"}).merge(bio["key_fangraphs"], on="key_fangraphs", how="right")

# genTeamFielding: returns aggregate team fielding stats for a season
def genTeamFielding(year):
    fieldtemp = pyb.team_fielding(year).set_index("teamIDfg").drop(columns={"Team"}).merge(pyb.team_batting(year)[["teamIDfg", "Team"]], on="teamIDfg", how="left")
    return fieldtemp[["teamIDfg", "Team", "G", "GS", "Inn", "PO", "A", "E", "FE", "TE", "DP", "Scp", "SB", "CS", "PB", "WP", "FP", "rSB", "rGDP", "rARM", "rGFP", "rPM", "DRS", "BIZ", "RZR", "OOZ", "ARM", "DPR","RngR", "ErrR","UZR","Def","FRM","OAA","RAA"]]
'''
def master(start_year, end_year):
    [b_fg, b_stat, b_bwar, b_team, b_bio, b_id] = genBatting(start_year, end_year)
    [p_fg, p_stat, p_bwar, p_team, p_bio, p_id] = genPitching(start_year, end_year)
    bio = pd.concat([b_bio, p_bio]).drop_duplicates(subset=['key_mlbam'])
    id = pd.concat([b_id, p_id]).drop_duplicates(subset=['key_mlbam'])
    gl = gameLogs(start_year, end_year)
    td = teamDepot(start_year, end_year)
    return [b_fg, p_fg, b_stat, p_stat, b_bwar, p_bwar, b_team, p_team, bio, id, gl, td]

