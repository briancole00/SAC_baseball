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

# genID: returns a dataframe with ID information of players in df
# expects a df with a FanGraphs ID
def genID(df):
    return df['Name'].reset_index().merge(pyb.chadwick_register(), on="key_fangraphs").drop(columns=['name_last','name_first'])

# genBio: returns a dataframe with biographical data for all players, including pitchers and hitters
def genBio(battingID, pitchingID):
    bio = pd.concat([battingID, pitchingID], ignore_index=True).drop_duplicates(subset=["key_fangraphs"]).merge(pyb.lahman.people().drop(columns=drop_bio).rename(columns={"playerID":"key_bbref","mlb_played_first":"debut","mlb_played_last":"most_recent_season"}),on="key_bbref",how="left")

    
# genBatting: generates basic hitting data
# qualified: 10 PA
def genBatting(year):
    team_dfs = []

    for team in fgteams:
        df_hittingtemp = pyb.batting_stats(year, team=team, qual=10)[['IDfg','Name','G'] + [col for col in hitting_cols]]
        df_hittingtemp['Team'] = fgteams[team]
        team_dfs.append(df_hittingtemp)

    hittingdf = pd.concat(team_dfs).rename(columns={"IDfg":"key_fangraphs"}).set_index("key_fangraphs")
    hittingdf = hittingdf[['Name','Team','G'] + [col for col in hitting_cols]]
    
    return hittingdf

# genPitching: generates basic pitching data
# qualified: 20 IP
def genPitching(year):
    team_dfs = []

    for team in fgteams:
        df_pitchtemp = pyb.pitching_stats(year, team=team, qual=20)[['IDfg', 'Name', 'G', 'GS'] + [col for col in pitching_cols]]
        df_pitchtemp['Team'] = fgteams[team]
        team_dfs.append(df_pitchtemp)

    pitchingdf = pd.concat(team_dfs).rename(columns={"IDfg":"key_fangraphs"}).set_index("key_fangraphs")
    pitchingdf = pitchingdf[['Name','Team','G','GS'] + [col for col in pitching_cols]]
    
    return pitchingdf

# genbWARBatting: returns a dataframe with bWAR batting data
# expects dataframe with mlbam ID (battingID)
def genbWARBatting(battingID, year):
    return pyb.bwar_bat(return_all=True).rename(columns={"mlb_ID":"key_mlbam"}).groupby(['year_ID']).get_group(year).merge(battingID['key_mlbam'].to_frame(), on="key_mlbam", how="right").drop_duplicates()

# genbWARpitching: returns a dataframe with bWAR pitching data
# expects a dataframe with mlbam ID (pitchingID)
def genbWARpitching(pitchingID, year):
    return pyb.bwar_pitch(return_all=True).rename(columns={"mlb_ID":"key_mlbam"}).groupby(['year_ID']).get_group(year).merge(pitchingID['key_mlbam'].to_frame(), on="key_mlbam", how="right").drop_duplicates()

# genStatcastHitting: generates a DataFrame with all Statcast hitting peripherals
def genStatcastBatting(battingID, year):
    return battingID['key_mlbam'].to_frame().merge(pyb.statcast_batter_exitvelo_barrels(year=year, minBBE=1).rename(columns={"player_id":"key_mlbam"}).drop(columns=["last_name","first_name"]), on="key_mlbam", how="left").merge(pyb.statcast_batter_expected_stats(year=year, minPA=10).rename(columns={"player_id":"key_mlbam"}).drop(columns=["last_name","first_name","year"]), on="key_mlbam",how="left")

# genStatcastPitching: generates a DataFrame with all Statcast pitching peripherals
def genStatcastPitching(pitchingID, year):
    return pitchingID['key_mlbam'].to_frame().merge(pyb.statcast_pitcher_exitvelo_barrels(year=year, minBBE=50).rename(columns={"player_id":"key_mlbam"}).drop(columns=["last_name","first_name"]), on="key_mlbam", how="left").merge(pyb.statcast_pitcher_expected_stats(year=year, minPA=50).rename(columns={"player_id":"key_mlbam"}).drop(columns=["last_name","first_name","year"]), on="key_mlbam",how="left").merge(pyb.statcast_pitcher_pitch_arsenal(year=year, minP=200, arsenal_type="avg_speed").rename(columns={"pitcher":"key_mlbam"}).drop(columns=["first_name", "last_name"]), on="key_mlbam", how="left").merge(pyb.statcast_pitcher_pitch_arsenal(year=year, minP=200, arsenal_type="avg_spin").rename(columns={"pitcher":"key_mlbam"}).drop(columns=["first_name", "last_name"]), on="key_mlbam", how="left")

# genFielding: generates fielding data
def genFielding(bio, year):
    return pyb.fielding_stats(year, qual=10).rename(columns={"IDfg":"key_fangraphs"}).set_index("key_fangraphs").drop(columns={"Name"}).merge(bio["key_fangraphs"], on="key_fangraphs", how="right")

# genTeamBatting: returns aggregate team batting stats for a season
def genTeamBatting(year):
    return pyb.team_batting(year)[['teamIDfg', 'Team'] + [col for col in hitting_cols]]

# genTeamPitching: returns aggregate team pitching stats for a season
def genTeamPitching(year):
    return pyb.team_pitching(year)[['teamIDfg', 'Team'] + [col for col in pitching_cols]]

# genTeamFielding: returns aggregate team fielding stats for a season
def genTeamFielding(year):
    fieldtemp = pyb.team_fielding(year).set_index("teamIDfg").drop(columns={"Team"}).merge(pyb.team_batting(year)[["teamIDfg", "Team"]], on="teamIDfg", how="left")
    return fieldtemp[["teamIDfg", "Team", "G", "GS", "Inn", "PO", "A", "E", "FE", "TE", "DP", "Scp", "SB", "CS", "PB", "WP", "FP", "rSB", "rGDP", "rARM", "rGFP", "rPM", "DRS", "BIZ", "RZR", "OOZ", "ARM", "DPR","RngR", "ErrR","UZR","Def","FRM","OAA","RAA"]]

def master(year):
    # generate basic hitting and pitching to build a playerbase for building the rest of the dataset
    batting_basic = genBatting(year)
    pitching_basic = genPitching(year)

    # generate master data using the basic data basis
    battingID = genID(batting_basic)
    pitchingID = genID(pitching_basic)
    batting_basic = batting_basic.merge(battingID[['key_fangraphs', 'key_mlbam']], on='key_fangraphs', how='right').drop_duplicates()
    pitching_basic = pitching_basic.merge(pitchingID[['key_fangraphs', 'key_mlbam']], on='key_fangraphs', how='right').drop_duplicates()
    bio = genBio(battingID, pitchingID)

    # generate additional dataframes
    bwarbatting = genbWARBatting(battingID,year)
    bwarpitching = genbWARpitching(pitchingID,year)
    statcastbatting = genStatcastBatting(battingID,year)
    statcastpitching = genStatcastPitching(pitchingID, year)
    fielding = genFielding(bio)

    # generate team dataframes
    teambatting = genTeamBatting(year)
    teampitching = genTeamPitching(year)
    teamfielding = genTeamFielding(year)

    return batting_basic, pitching_basic, fielding, bio, bwarbatting, bwarpitching, statcastbatting, statcastpitching, teambatting, teampitching, teamfielding