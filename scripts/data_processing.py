import pandas as pd
import numpy as np

def load_and_clean_data(groups_path, matches_path, history_path):
    """
    Loads and cleans the initial datasets from the 'data' folder.
    Returns cleaned DataFrames for groups, matches, and historical data.
    """
    # 1. Load Groups Data
    df_groups = pd.read_csv(groups_path, header=None)
    df_groups.columns = df_groups.iloc[0]
    df_groups = df_groups.iloc[1:].reset_index(drop=True)
    df_groups = df_groups[['Group', 'Team', 'Ranking Points']]
    # Clean team names by removing ranking numbers e.g., 'Brazil (1)' -> 'Brazil'
    df_groups['Team'] = df_groups['Team'].str.replace(r'\s*\(\d+\)', '', regex=True)
    df_groups['Ranking Points'] = df_groups['Ranking Points'].astype(float)
    
    # 2. Load Matches Data
    df_matches = pd.read_csv(matches_path)
    df_matches['Team 1'] = df_matches['Team 1'].str.replace(r'\s*\(\d+\)', '', regex=True)
    df_matches['Team 2'] = df_matches['Team 2'].str.replace(r'\s*\(\d+\)', '', regex=True)
    df_matches['Match No.'] = pd.to_numeric(df_matches['Match No.'], errors='coerce')
    
    # 3. Load Historical Match Data
    df_history = pd.read_csv(history_path)
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history = df_history.dropna(subset=['home_score', 'away_score'])
    
    # Filter for 'Modern Football' era (1994 onwards)
    df_hist_modern = df_history[df_history['date'] >= '1994-01-01'].copy()
    
    return df_groups, df_matches, df_hist_modern, df_history

def get_team_stats(team_name, df_hist, n_recent):
    """
    Calculates Historical Attack/Defense and Recent Form based on 'n_recent' matches.
    This makes the team's form dynamic based on user selection.
    """
    # Get all matches for the specific team and sort from newest to oldest
    team_matches = df_hist[(df_hist['home_team'] == team_name) | (df_hist['away_team'] == team_name)].sort_values('date', ascending=False)
    
    if team_matches.empty:
        return pd.Series({'Hist_Attack': 1.0, 'Hist_Defense': 1.0, 'Recent_Form': 1.0, 'WC_Experience': 0})
        
    # --- A. Historical Attack & Defense (Overall modern era) ---
    goals_scored = np.where(team_matches['home_team'] == team_name, team_matches['home_score'], team_matches['away_score'])
    goals_conceded = np.where(team_matches['home_team'] == team_name, team_matches['away_score'], team_matches['home_score'])
    
    hist_attack = goals_scored.mean() if len(goals_scored) > 0 else 1.0
    hist_defense = goals_conceded.mean() if len(goals_conceded) > 0 else 1.0
    
    # --- B. Recent Form (Based on user-selected 'n_recent' matches) ---
    recent_matches = team_matches.head(n_recent)
    recent_scored = np.where(recent_matches['home_team'] == team_name, recent_matches['home_score'], recent_matches['away_score'])
    recent_conceded = np.where(recent_matches['home_team'] == team_name, recent_matches['away_score'], recent_matches['home_score'])
    
    # Calculate Recent Goal Difference (GD)
    recent_gd = (recent_scored - recent_conceded).mean() if len(recent_scored) > 0 else 0
    # Convert GD into a multiplier (e.g., +1 GD gives a 1.1x boost, bounded between 0.5 and 1.5)
    recent_form_multiplier = 1 + (recent_gd * 0.1) 
    recent_form_multiplier = max(0.5, min(1.5, recent_form_multiplier)) 
    
    # --- C. World Cup Experience ---
    wc_matches = team_matches[team_matches['tournament'] == 'FIFA World Cup']
    wc_exp = len(wc_matches)
    
    return pd.Series({
        'Hist_Attack': hist_attack, 
        'Hist_Defense': hist_defense, 
        'Recent_Form': recent_form_multiplier, 
        'WC_Experience': wc_exp
    })

def prepare_features(df_groups, df_hist_modern, n_recent):
    """
    Applies team stats to all teams and standardizes variables like WC Experience.
    """
    df_teams = df_groups.copy()
    
    # Generate stats for all teams dynamically
    stats = df_teams['Team'].apply(lambda x: get_team_stats(x, df_hist_modern, n_recent))
    df_teams = pd.concat([df_teams, stats], axis=1)
    
    # Normalize World Cup Experience to be a value between 0 and 1
    max_exp = df_teams['WC_Experience'].max()
    if max_exp > 0:
        df_teams['WC_Experience_Norm'] = df_teams['WC_Experience'] / max_exp
    else:
        df_teams['WC_Experience_Norm'] = 0
        
    return df_teams
