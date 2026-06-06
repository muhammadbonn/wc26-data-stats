import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_tournament_winners(df_champs):
    """
    Creates an interactive Plotly bar chart for tournament win probabilities.
    Keeps the same structure but fully in English.
    """
    df_top = df_champs.head(16).copy()
    
    # Sort ascending for Plotly horizontal bar chart so the highest is at the top
    df_top = df_top.sort_values(by='Champion_Prob', ascending=True)
    
    fig = px.bar(
        df_top, 
        x='Champion_Prob', 
        y='Team', 
        orientation='h',
        title='🏆 Probability of Winning World Cup 2026 (Top 16 Teams)',
        labels={'Champion_Prob': 'Win Probability (%)', 'Team': 'Team'},
        color='Champion_Prob',
        color_continuous_scale='viridis',
        text='Champion_Prob'
    )
    
    # Format the text to show one decimal place
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(template='plotly_white', height=600)
    
    return fig

def plot_advancing_teams(df_adv):
    """
    Creates an interactive Plotly chart for top 32 teams grouped by Power Tiers.
    """
    df_top32 = df_adv.head(32).copy()
    
    # Assign Tier Names (8 teams per tier) just like the Notebook
    tier_names = [
        'Tier 1: The Favorites', 
        'Tier 2: The Challengers', 
        'Tier 3: The Dark Horses', 
        'Tier 4: The Contenders'
    ]
    df_top32['Tier'] = np.repeat(tier_names, 8)
    
    # Reverse the dataframe so Tier 1 is plotted at the very top
    df_top32 = df_top32.iloc[::-1].reset_index(drop=True)
    
    # Define custom colors for the tiers
    tier_colors = {
        'Tier 1: The Favorites': '#2E8B57',    # Sea Green
        'Tier 2: The Challengers': '#4682B4',  # Steel Blue
        'Tier 3: The Dark Horses': '#DAA520',  # Goldenrod
        'Tier 4: The Contenders': '#CD5C5C'    # Indian Red
    }
    
    fig = px.bar(
        df_top32,
        x='Advancement_Prob',
        y='Team',
        color='Tier',
        orientation='h',
        title='📊 Top 32 Qualified Teams (Round of 32) - Power Tiers',
        labels={'Advancement_Prob': 'Advancement Probability (%)', 'Team': '', 'Tier': 'Power Tier'},
        color_discrete_map=tier_colors,
        height=850
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': df_top32['Team'].tolist()}, 
        template='plotly_white',
        showlegend=True
    )
    return fig

def plot_exciting_matches(df_preds):
    """
    Identifies the most closely contested matches and plots them as a Stacked Bar Chart.
    Shows the probability breakdown: Team 1 Win | Draw | Team 2 Win.
    """
    df_exciting = df_preds.copy()
    
    # Calculate closeness: smaller absolute difference means a highly contested match
    df_exciting['Win_Diff'] = abs(df_exciting['T1_Win_%'] - df_exciting['T2_Win_%'])
    
    # Sort by the smallest win difference first, then highest draw percentage
    df_top = df_exciting.sort_values(by=['Win_Diff', 'Draw_%'], ascending=[True, False]).head(10)
    
    # Create a unified match name label
    df_top['Match_Name'] = df_top['Team 1'] + " vs " + df_top['Team 2']
    
    # Reverse so the most exciting match is at the top of the chart
    df_top = df_top.iloc[::-1].reset_index(drop=True)
    
    # Build the Stacked Bar Chart
    fig = go.Figure()
    
    # Trace 1: Team 1 Win Probability
    fig.add_trace(go.Bar(
        y=df_top['Match_Name'],
        x=df_top['T1_Win_%'],
        name='Team 1 Win',
        orientation='h',
        marker=dict(color='#1f77b4'), # Blue
        text=df_top['T1_Win_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Trace 2: Draw Probability
    fig.add_trace(go.Bar(
        y=df_top['Match_Name'],
        x=df_top['Draw_%'],
        name='Draw',
        orientation='h',
        marker=dict(color='#7f7f7f'), # Gray
        text=df_top['Draw_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Trace 3: Team 2 Win Probability
    fig.add_trace(go.Bar(
        y=df_top['Match_Name'],
        x=df_top['T2_Win_%'],
        name='Team 2 Win',
        orientation='h',
        marker=dict(color='#d62728'), # Red
        text=df_top['T2_Win_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='🔥 Top 10 Most Closely Contested Matches',
        xaxis_title='Probability (%)',
        yaxis_title='',
        template='plotly_white',
        height=600,
        legend_title='Match Outcome',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_all_matches(df_preds):
    """
    Creates a tall Stacked Bar Chart showing the predictions for ALL group stage matches.
    """
    df_all = df_preds.copy()
    
    # Sort descending so Match No. 1 appears at the very top of the Plotly horizontal chart
    df_all = df_all.sort_values(by='Match No.', ascending=False).reset_index(drop=True)
    
    # Create a nice label combining Match Number and Teams
    df_all['Match_Name'] = "M" + df_all['Match No.'].astype(int).astype(str) + ": " + df_all['Team 1'] + " vs " + df_all['Team 2']
    
    # Build the Stacked Bar Chart
    fig = go.Figure()
    
    # Trace 1: Team 1 Win Probability
    fig.add_trace(go.Bar(
        y=df_all['Match_Name'],
        x=df_all['T1_Win_%'],
        name='Team 1 Win',
        orientation='h',
        marker=dict(color='#1f77b4'), # Blue
        text=df_all['T1_Win_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Trace 2: Draw Probability
    fig.add_trace(go.Bar(
        y=df_all['Match_Name'],
        x=df_all['Draw_%'],
        name='Draw',
        orientation='h',
        marker=dict(color='#7f7f7f'), # Gray
        text=df_all['Draw_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Trace 3: Team 2 Win Probability
    fig.add_trace(go.Bar(
        y=df_all['Match_Name'],
        x=df_all['T2_Win_%'],
        name='Team 2 Win',
        orientation='h',
        marker=dict(color='#d62728'), # Red
        text=df_all['T2_Win_%'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='All Group Stage Matches Predictions',
        xaxis_title='Probability (%)',
        yaxis_title='',
        template='plotly_white',
        height=2000, # Made it very tall so all 72 matches fit comfortably
        legend_title='Match Outcome',
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1)
    )
    
    return fig
