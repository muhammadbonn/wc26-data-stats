import plotly.express as px
import pandas as pd

def plot_tournament_winners(df_champs):
    """
    Creates an interactive Plotly bar chart for tournament win probabilities.
    """
    df_top = df_champs.head(16).copy()
    
    fig = px.bar(
        df_top, 
        x='Champion_Prob', 
        y='Team', 
        orientation='h',
        title='🏆 إحتمالية الفوز بكأس العالم 2026 (أعلى 16 فريق)',
        labels={'Champion_Prob': 'نسبة الفوز (%)', 'Team': 'الفريق'},
        color='Champion_Prob',
        color_continuous_scale='viridis',
        text='Champion_Prob'
    )
    
    # Format the text to show one decimal place
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
    
    return fig

def plot_advancing_teams(df_adv):
    """
    Creates an interactive Plotly chart for top 32 teams advancing out of groups.
    """
    df_top32 = df_adv.head(32).copy()
    
    fig = px.bar(
        df_top32,
        x='Advancement_Prob',
        y='Team',
        orientation='h',
        title='📊 إحتمالية التأهل لدور الـ 32',
        labels={'Advancement_Prob': 'نسبة التأهل (%)', 'Team': 'الفريق'},
        color='Advancement_Prob',
        color_continuous_scale='Blues',
        height=800
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
    return fig

def get_exciting_matches(df_preds):
    """
    Identifies the most closely contested matches in the group stage.
    Calculates excitement based on high draw probabilities and lowest Win % difference.
    """
    df_exciting = df_preds.copy()
    
    # Difference in win probability; smaller = closer match
    df_exciting['Win_Diff'] = abs(df_exciting['T1_Win_%'] - df_exciting['T2_Win_%'])
    
    # Sort by closest win difference, then highest draw probability
    df_top_exciting = df_exciting.sort_values(by=['Win_Diff', 'Draw_%'], ascending=[True, False]).head(15)
    
    # Clean up the output table for the user interface
    res = df_top_exciting[['Match No.', 'Team 1', 'Team 2', 'T1_Win_%', 'Draw_%', 'T2_Win_%']].copy()
    
    # Rename columns for presentation
    res.columns = ['رقم المباراة', 'الفريق الأول', 'الفريق الثاني', 'فوز الفريق 1 (%)', 'تعادل (%)', 'فوز الفريق 2 (%)']
    res = res.reset_index(drop=True)
    res.index += 1
    
    return res
