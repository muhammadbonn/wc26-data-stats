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
st.title("🏆 محاكي كأس العالم 2026 التفاعلي")
st.markdown("""
هذا التطبيق يتيح لك توقع الفائز بكأس العالم 2026 بناءً على نموذج رياضي (Poisson Distribution) ومحاكاة (Monte Carlo).
**تحكم في الأوزان والمعايير من القائمة الجانبية لبناء محاكاتك الخاصة!**
""")

# ================= Sidebar Configuration =================
st.sidebar.header("⚙️ إعدادات محرك التوقع")

# Slider for recent matches count (Form Decay)
n_recent = st.sidebar.slider(
    "عدد المباريات الأخيرة (لتقييم الفورمة الحالية):", 
    min_value=3, max_value=20, value=10, step=1,
    help="كم عدد المباريات السابقة التي تريد أخذها في الاعتبار لحساب فورمة الفريق الحالية؟"
)

# Slider for Expected Goals (xG) weight
xg_weight = st.sidebar.slider(
    "وزن الأهداف المتوقعة (xG) التاريخي:", 
    min_value=0.1, max_value=1.0, value=0.6, step=0.1,
    help="قيمة 0.6 تعني الاعتماد بنسبة 60% على التاريخ و 40% على الفورمة الحالية."
)

# Slider for World Cup Experience weight
wc_exp_weight = st.sidebar.slider(
    "تأثير خبرة كأس العالم:", 
    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
    help="نسبة زيادة قوة الفريق بناءً على تاريخه ومشاركاته السابقة في كأس العالم."
)

# Selectbox for number of Monte Carlo simulations
n_sims = st.sidebar.selectbox(
    "عدد مرات محاكاة البطولة (Monte Carlo):", 
    options=[1000, 5000, 10000], index=0,
    help="رقم أكبر يعطي نتائج أدق لكنه يستغرق وقتاً أطول في الحساب."
)

# Define file paths for the datasets
DATA_DIR = "data"
GROUPS_PATH = os.path.join(DATA_DIR, "wc26_groups.csv")
MATCHES_PATH = os.path.join(DATA_DIR, "wc26_matches.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "all_results.csv")

# ================= Simulation Execution =================
# Trigger the pipeline when the user clicks the button
if st.sidebar.button("🚀 ابدأ التوقع والمحاكاة", type="primary"):
    
    # Check if the necessary data files exist in the 'data' directory
    if not (os.path.exists(GROUPS_PATH) and os.path.exists(MATCHES_PATH) and os.path.exists(HISTORY_PATH)):
        st.error("⚠️ ملفات البيانات غير موجودة! يرجى التأكد من وجود مجلد 'data' وبداخله الملفات المطلوبة.")
    else:
        with st.spinner("جاري تحميل البيانات وتحليل الإحصائيات... ⏳"):
            # 1. Load and clean the raw data
            df_groups, df_matches, df_hist_modern, df_history = load_and_clean_data(GROUPS_PATH, MATCHES_PATH, HISTORY_PATH)
            
            # 2. Prepare team features and calculate dynamic form
            df_teams = prepare_features(df_groups, df_hist_modern, n_recent)
            
        with st.spinner("جاري حساب توقعات المباريات باستخدام توزيع بواسون... 🧮"):
            # 3. Generate predictions using the Poisson model and user weights
            df_preds = generate_tournament_predictions(df_matches, df_teams, xg_weight, wc_exp_weight)
            
        with st.spinner(f"جاري تشغيل محاكاة دور المجموعات {n_sims} مرة... 🎲"):
            # 4. Run Monte Carlo simulation for the group stage
            df_adv = simulate_group_stage(df_preds, df_teams, n_sims)
            
        with st.spinner(f"جاري تشغيل محاكاة الأدوار الإقصائية وتحديد البطل... 🏆"):
            # 5. Run Monte Carlo simulation for the knockout rounds
            df_champs = simulate_knockout_stage(df_adv, df_teams, n_sims)
            
        st.success("✅ تمت المحاكاة بنجاح!")
        
        # ================= Display Results (UI Tabs) =================
        tab1, tab2, tab3 = st.tabs(["🏆 أبطال العالم المتوقعين", "📊 المتأهلين لدور الـ 32", "🔥 المباريات الحماسية"])
        
        # Tab 1: Tournament Winner Probabilities
        with tab1:
            st.subheader("فرص الفرق في الفوز بكأس العالم 2026")
            fig_champs = plot_tournament_winners(df_champs)
            st.plotly_chart(fig_champs, use_container_width=True)
            
        # Tab 2: Top 32 Advancing Teams
        with tab2:
            st.subheader("أكثر الفرق حظوظاً لتخطي دور المجموعات")
            fig_adv = plot_advancing_teams(df_adv)
            st.plotly_chart(fig_adv, use_container_width=True)
            
        # Tab 3: Most Exciting Group Stage Matches
        with tab3:
            st.subheader("أكثر مباريات دور المجموعات إثارة وتقارباً في المستوى")
            df_exciting = get_exciting_matches(df_preds)
            st.dataframe(df_exciting, use_container_width=True, hide_index=True)

else:
    # Prompt the user to start the simulation if the button hasn't been clicked yet
    st.info("👈 قم بتعديل الأوزان من القائمة الجانبية ثم اضغط على زر 'ابدأ التوقع والمحاكاة' لرؤية النتائج.")
