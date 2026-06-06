import pandas as pd
import numpy as np

def simulate_group_stage(df_preds, df_teams, n_sims):
    """
    Runs Monte Carlo simulations on the group stage matches.
    Tracks how many times each team advances based on Match Probabilities.
    """
    team_groups = dict(zip(df_teams['Team'], df_teams['Group']))
    advancement_tracker = {team: 0 for team in team_groups.keys()}
    
    # Extract probabilities for fast simulation loops
    matches = df_preds[['Team 1', 'Team 2', 'T1_Win_%', 'Draw_%', 'T2_Win_%']].values
    
    for _ in range(n_sims):
        # Dictionary to track points for one single simulation
        sim_points = {team: 0 for team in team_groups.keys()}
        
        for m in matches:
            t1, t2, w1, draw, w2 = m
            # Roll a random number between 0 and 100
            roll = np.random.uniform(0, 100)
            
            # Distribute points based on the probability roll
            if roll < w1:
                sim_points[t1] += 3
            elif roll < w1 + draw:
                sim_points[t1] += 1
                sim_points[t2] += 1
            else:
                sim_points[t2] += 3
                
        # Determine group standings
        df_sim = pd.DataFrame(list(sim_points.items()), columns=['Team', 'Points'])
        df_sim['Group'] = df_sim['Team'].map(team_groups)
        
        # Sort by points within groups
        df_sim = df_sim.sort_values(by=['Group', 'Points'], ascending=[True, False])
        
        # Rank teams within their groups (1 to 4)
        df_sim['Rank'] = df_sim.groupby('Group').cumcount() + 1
        
        # In 2026, Top 2 teams from 12 groups + 8 best third-place teams advance
        top_2 = df_sim[df_sim['Rank'] <= 2]['Team'].tolist()
        best_thirds = df_sim[df_sim['Rank'] == 3].sort_values(by='Points', ascending=False).head(8)['Team'].tolist()
        
        advancing_teams = top_2 + best_thirds
        
        # Update tracker
        for team in advancing_teams:
            advancement_tracker[team] += 1
            
    # Calculate final advancement percentages
    df_adv = pd.DataFrame(list(advancement_tracker.items()), columns=['Team', 'Total_Adv'])
    df_adv['Advancement_Prob'] = (df_adv['Total_Adv'] / n_sims) * 100
    return df_adv.sort_values(by='Advancement_Prob', ascending=False)

def simulate_knockout_stage(df_adv, df_teams, n_sims):
    """
    Simulates the knockout stage (Round of 32 to Final) to find the tournament winner.
    """
    # Get the top 32 most likely teams to advance
    top_32_teams = df_adv.head(32)['Team'].tolist()
    champions = {team: 0 for team in top_32_teams}
    
    # Calculate a simple "Power Score" for each team for the KO logic
    df_teams['Power_Score'] = df_teams['Hist_Attack'] * df_teams['Recent_Form'] / df_teams['Hist_Defense']
    team_strengths = dict(zip(df_teams['Team'], df_teams['Power_Score']))
    
    for _ in range(n_sims):
        current_round = list(top_32_teams)
        
        # Loop until 1 winner remains
        while len(current_round) > 1:
            next_round = []
            # Play matches in pairs (Simulated Bracket)
            for i in range(0, len(current_round), 2):
                if i+1 >= len(current_round): break
                t1, t2 = current_round[i], current_round[i+1]
                
                s1 = team_strengths.get(t1, 1.0)
                s2 = team_strengths.get(t2, 1.0)
                
                # Probability of Team 1 winning based on relative power scores
                prob_t1 = s1 / (s1 + s2)
                
                if np.random.random() < prob_t1:
                    next_round.append(t1)
                else:
                    next_round.append(t2)
                    
            current_round = next_round
            
        # Add a win to the overall champion of this simulation
        if current_round:
            champions[current_round[0]] += 1
        
    df_champs = pd.DataFrame(list(champions.items()), columns=['Team', 'Win_Count'])
    df_champs['Champion_Prob'] = (df_champs['Win_Count'] / n_sims) * 100
    return df_champs.sort_values(by='Champion_Prob', ascending=False)
