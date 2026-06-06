import streamlit as st
import pandas as pd
import os

# Import custom modules from the 'scripts' directory
from scripts.data_processing import load_and_clean_data, prepare_features
from scripts.model import generate_tournament_predictions
from scripts.simulation import simulate_group_stage, simulate_knockout_stage
from scripts.visualization import plot_tournament_winners, plot_advancing_teams, get_exciting_matches

# Configure the Streamlit page layout and metadata
st.set_page_config(page_title="World Cup 2026 Simulator", page_icon="🏆", layout="wide")

# Main Title and Description
st.title("🏆 Interactive World Cup 2026 Simulator")
st.markdown("""
This application allows you to predict the winner of the 2026 World Cup based on a mathematical model (Poisson Distribution) and simulation (Monte Carlo).
**Adjust the weights and parameters from the sidebar to build your own custom simulation!**
""")

# ================= Sidebar Configuration =================
st.sidebar.header("⚙️ Prediction Engine Settings")

# Slider for recent matches count (Form Decay)
n_recent = st.sidebar.slider(
    "Number of recent matches (Current Form):", 
    min_value=3, max_value=20, value=10, step=1,
    help="How many previous matches do you want to consider to calculate the team's current form?"
)

# Slider for Expected Goals (xG) weight
xg_weight = st.sidebar.slider(
    "Historical Expected Goals (xG) Weight:", 
    min_value=0.1, max_value=1.0, value=0.6, step=0.1,
    help="A value of 0.6 means relying 60% on historical data and 40% on current form."
)

# Slider for World Cup Experience weight
wc_exp_weight = st.sidebar.slider(
    "World Cup Experience Impact:", 
    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
    help="The percentage increase in a team's strength based on their history and previous World Cup participations."
)

# Selectbox for number of Monte Carlo simulations
n_sims = st.sidebar.selectbox(
    "Number of Monte Carlo simulations:", 
    options=[1000, 5000, 10000], index=0,
    help="A higher number provides more accurate results but takes longer to compute."
)

# Define file paths for the datasets
DATA_DIR = "data"
GROUPS_PATH = os.path.join(DATA_DIR, "wc26_groups.csv")
MATCHES_PATH = os.path.join(DATA_DIR, "wc26_matches.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "all_results.csv")

# ================= Simulation Execution =================
# Trigger the pipeline when the user clicks the button
if st.sidebar.button("🚀 Start Prediction & Simulation", type="primary"):
    
    # Check if the necessary data files exist in the 'data' directory
    if not (os.path.exists(GROUPS_PATH) and os.path.exists(MATCHES_PATH) and os.path.exists(HISTORY_PATH)):
        st.error("⚠️ Data files not found! Please ensure the 'data' folder exists and contains the required files.")
    else:
        with st.spinner("Loading data and analyzing statistics... ⏳"):
            # 1. Load and clean the raw data
            df_groups, df_matches, df_hist_modern, df_history = load_and_clean_data(GROUPS_PATH, MATCHES_PATH, HISTORY_PATH)
            
            # 2. Prepare team features and calculate dynamic form
            df_teams = prepare_features(df_groups, df_hist_modern, n_recent)
            
        with st.spinner("Calculating match predictions using Poisson distribution... 🧮"):
            # 3. Generate predictions using the Poisson model and user weights
            df_preds = generate_tournament_predictions(df_matches, df_teams, xg_weight, wc_exp_weight)
            
        with st.spinner(f"Running group stage simulation {n_sims} times... 🎲"):
            # 4. Run Monte Carlo simulation for the group stage
            df_adv = simulate_group_stage(df_preds, df_teams, n_sims)
            
        with st.spinner(f"Running knockout stage simulation and determining the champion... 🏆"):
            # 5. Run Monte Carlo simulation for the knockout rounds
            df_champs = simulate_knockout_stage(df_adv, df_teams, n_sims)
            
        st.success("✅ Simulation completed successfully!")
        
        # ================= Display Results (UI Tabs) =================
        tab1, tab2, tab3 = st.tabs(["🏆 Predicted Champions", "📊 Top 32 Qualifiers", "🔥 Exciting Matches"])
        
        # Tab 1: Tournament Winner Probabilities
        with tab1:
            st.subheader("Teams' Chances to Win the 2026 World Cup")
            fig_champs = plot_tournament_winners(df_champs)
            st.plotly_chart(fig_champs, use_container_width=True)
            
        # Tab 2: Top 32 Advancing Teams
        with tab2:
            st.subheader("Teams Most Likely to Advance from the Group Stage")
            fig_adv = plot_advancing_teams(df_adv)
            st.plotly_chart(fig_adv, use_container_width=True)
            
        # Tab 3: Most Exciting Group Stage Matches
        with tab3:
            st.subheader("Most Exciting and Closely Contested Group Stage Matches")
            df_exciting = get_exciting_matches(df_preds)
            st.dataframe(df_exciting, use_container_width=True, hide_index=True)

else:
    # Prompt the user to start the simulation if the button hasn't been clicked yet
    st.info("👈 Adjust the parameters from the sidebar, then click 'Start Prediction & Simulation' to view the results.")
