import numpy as np
import pandas as pd
from scipy.stats import poisson

def predict_match(team1, team2, df_teams, xg_weight, wc_exp_weight):
    """
    Predicts match outcome probabilities using Poisson distribution.
    Incorporates dynamic weights (xg_weight, wc_exp_weight) defined by the user.
    """
    t1_data = df_teams[df_teams['Team'] == team1]
    t2_data = df_teams[df_teams['Team'] == team2]
    
    # Fallback if team is not found in data
    if t1_data.empty or t2_data.empty:
        return 33.3, 33.4, 33.3, 1.0, 1.0
        
    t1 = t1_data.iloc[0]
    t2 = t2_data.iloc[0]
    
    GLOBAL_AVG = 1.3 # Average goals scored per team in a match
    
    # --- 1. Base Expected Goals (xG) Calculation ---
    # T1 Attack strength vs T2 Defense weakness
    base_xg1 = t1['Hist_Attack'] * t2['Hist_Defense'] * GLOBAL_AVG
    base_xg2 = t2['Hist_Attack'] * t1['Hist_Defense'] * GLOBAL_AVG
    
    # --- 2. Apply Recent Form & xG Weight (User Input) ---
    # The user decides how much to rely on historical xG vs Recent Form.
    # If xg_weight is 0.7, then form_weight is 0.3
    form_weight = 1.0 - xg_weight
    
    # Modifying xG based on user's preference for recent form
    xg1 = base_xg1 * (xg_weight + (t1['Recent_Form'] * form_weight))
    xg2 = base_xg2 * (xg_weight + (t2['Recent_Form'] * form_weight))
    
    # --- 3. Apply World Cup Experience Multiplier (User Input) ---
    # Boost xG for teams with extensive WC history, based on user's chosen weight
    xg1 *= (1 + (t1['WC_Experience_Norm'] * wc_exp_weight))
    xg2 *= (1 + (t2['WC_Experience_Norm'] * wc_exp_weight))
    
    # --- 4. Poisson Distribution to Calculate Probabilities ---
    max_goals = 10 # Cap the calculation at 10 goals per match for efficiency
    
    # Generate arrays of probabilities for scoring 0 to 9 goals for both teams
    team_pred = [[poisson.pmf(i, team_avg) for i in range(max_goals)] for team_avg in [xg1, xg2]]
    
    # Create the probability matrix (Team 1 goals vs Team 2 goals)
    prob_matrix = np.outer(team_pred[0], team_pred[1])
    
    # Calculate final match outcomes:
    # Sum of lower triangle = Team 1 wins
    # Sum of diagonal = Draw
    # Sum of upper triangle = Team 2 wins
    t1_win_prob = np.sum(np.tril(prob_matrix, -1)) * 100
    draw_prob = np.sum(np.diag(prob_matrix)) * 100
    t2_win_prob = np.sum(np.triu(prob_matrix, 1)) * 100
    
    return t1_win_prob, draw_prob, t2_win_prob, xg1, xg2

def generate_tournament_predictions(df_matches, df_teams, xg_weight, wc_exp_weight):
    """
    Iterates through all group stage matches and generates predictions.
    """
    df_preds = df_matches.copy()
    
    # Filter for Group Stage matches only (assuming first 72 matches in 2026 format)
    df_preds = df_preds[df_preds['Match No.'] <= 72].copy()
    
    results = []
    for index, row in df_preds.iterrows():
        t1 = row['Team 1']
        t2 = row['Team 2']
        
        # Predict outcome using our dynamic Poisson model
        w1, draw, w2, xg1, xg2 = predict_match(t1, t2, df_teams, xg_weight, wc_exp_weight)
        
        results.append({
            'T1_xG': round(xg1, 2),
            'T2_xG': round(xg2, 2),
            'T1_Win_%': round(w1, 1),
            'Draw_%': round(draw, 1),
            'T2_Win_%': round(w2, 1)
        })
        
    df_results = pd.DataFrame(results, index=df_preds.index)
    return pd.concat([df_preds, df_results], axis=1)
