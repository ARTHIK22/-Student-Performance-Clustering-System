import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import save_sample_dataset, generate_student_dataset
from utils.clustering import (
    preprocess_data,
    compute_elbow_and_silhouette,
    fit_kmeans,
    get_unscaled_centroids,
    assign_personas,
    predict_single_student
)
from utils.visualization import (
    plot_elbow_and_silhouette,
    plot_2d_clusters,
    plot_3d_clusters,
    plot_feature_boxplots,
    plot_correlation_matrix
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Clustering System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME CSS ────────────────────────────────────────────────────────────────
def get_theme_css(dark: bool) -> str:
    if dark:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Base ── */
.stApp, .main, section.main > div.block-container {
    background: linear-gradient(135deg, #0A0F1E 0%, #0F172A 50%, #0D1B2A 100%) !important;
    min-height: 100vh;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #312E81 50%, #0F172A 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px -10px rgba(99,102,241,0.3);
}
.app-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.app-header-icon { font-size: 3.2rem; line-height: 1; margin-bottom: 0.6rem; }
.app-header-title {
    font-size: 2.4rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.4rem;
}
.app-header-sub { font-size: 1rem; color: #94A3B8; font-weight: 400; }
.app-header-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.4);
    color: #A5B4FC; font-size: 0.75rem; font-weight: 600;
    padding: 0.3rem 0.85rem; border-radius: 20px; margin-top: 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1F35 0%, #0F172A 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1E293B 0%, #1A2744 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    border-radius: 16px 16px 0 0;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(99,102,241,0.25); }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.metric-title { font-size: 0.78rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2rem; font-weight: 800; color: #E2E8F0; margin-top: 0.2rem; }
.metric-sub { font-size: 0.75rem; color: #475569; margin-top: 0.15rem; }

/* ── Persona cards ── */
.persona-card {
    background: linear-gradient(135deg, #1E293B 0%, #1A2744 100%);
    border-left: 4px solid #818CF8;
    border-top: 1px solid rgba(99,102,241,0.2);
    border-right: 1px solid rgba(99,102,241,0.1);
    border-bottom: 1px solid rgba(99,102,241,0.1);
    border-radius: 0 12px 12px 0;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    transition: transform 0.2s, box-shadow 0.2s;
}
.persona-card:hover { transform: translateX(4px); box-shadow: 0 6px 20px rgba(99,102,241,0.2); }
.persona-title { font-size: 1.1rem; font-weight: 700; color: #E2E8F0; }
.persona-count { font-size: 1.6rem; font-weight: 800; color: #818CF8; margin: 0.3rem 0; }
.persona-pct { font-size: 0.85rem; font-weight: 500; color: #64748B; }
.persona-desc { font-size: 0.9rem; color: #94A3B8; margin-top: 0.4rem; line-height: 1.5; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.2rem; font-weight: 700; color: #E2E8F0;
    margin: 1.5rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,41,59,0.5) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
}

/* ── Misc ── */
h1, h2, h3, h4 { color: #F1F5F9 !important; }
p, .stMarkdown { color: #CBD5E1 !important; }
.stDataFrame { border-radius: 12px !important; border: 1px solid rgba(99,102,241,0.2) !important; }
.stAlert { border-radius: 12px !important; }
.stForm { background: rgba(30,41,59,0.5) !important; border: 1px solid rgba(99,102,241,0.2) !important; border-radius: 16px !important; }
.stNumberInput input, .stTextInput input { background: #1E293B !important; color: #E2E8F0 !important; border-color: rgba(99,102,241,0.3) !important; }
.theme-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    color: #A5B4FC; font-size: 0.78rem; font-weight: 600;
    padding: 0.25rem 0.7rem; border-radius: 20px; margin-bottom: 0.5rem;
}
.predictor-result-card {
    background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
    border: 1px solid rgba(16,185,129,0.4); border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 8px 24px rgba(16,185,129,0.2);
}
.advisor-card {
    background: rgba(30,41,59,0.7); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px; padding: 1.5rem;
}
</style>
"""
    else:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Base ── */
.stApp, .main, section.main > div.block-container {
    background: linear-gradient(135deg, #F0F4FF 0%, #F8FAFC 50%, #EFF6FF 100%) !important;
    min-height: 100vh;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1E293B 0%, #1E3A8A 50%, #312E81 100%);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
    box-shadow: 0 20px 60px -10px rgba(30,58,138,0.35);
}
.app-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.2) 0%, transparent 60%);
    pointer-events: none;
}
.app-header-icon { font-size: 3.2rem; line-height: 1; margin-bottom: 0.6rem; }
.app-header-title {
    font-size: 2.4rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FFFFFF 0%, #BAE6FD 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.4rem;
}
.app-header-sub { font-size: 1rem; color: #94A3B8; font-weight: 400; }
.app-header-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
    color: #BAE6FD; font-size: 0.75rem; font-weight: 600;
    padding: 0.3rem 0.85rem; border-radius: 20px; margin-top: 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%) !important;
    border-right: 1px solid #E2E8F0;
    box-shadow: 2px 0 20px rgba(99,102,241,0.05);
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 1px solid #E8EDFF;
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    border-radius: 16px 16px 0 0;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(99,102,241,0.15); }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.metric-title { font-size: 0.78rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2rem; font-weight: 800; color: #0F172A; margin-top: 0.2rem; }
.metric-sub { font-size: 0.75rem; color: #94A3B8; margin-top: 0.15rem; }

/* ── Persona cards ── */
.persona-card {
    background: white;
    border-left: 4px solid #6366F1;
    border-top: 1px solid #E8EDFF;
    border-right: 1px solid #E8EDFF;
    border-bottom: 1px solid #E8EDFF;
    border-radius: 0 12px 12px 0;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}
.persona-card:hover { transform: translateX(4px); box-shadow: 0 6px 20px rgba(99,102,241,0.12); }
.persona-title { font-size: 1.1rem; font-weight: 700; color: #1E293B; }
.persona-count { font-size: 1.6rem; font-weight: 800; color: #4F46E5; margin: 0.3rem 0; }
.persona-pct { font-size: 0.85rem; font-weight: 500; color: #94A3B8; }
.persona-desc { font-size: 0.9rem; color: #64748B; margin-top: 0.4rem; line-height: 1.5; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.2rem; font-weight: 700; color: #1E293B;
    margin: 1.5rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #E8EDFF;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(99,102,241,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}

/* ── Misc ── */
.stDataFrame { border-radius: 12px !important; border: 1px solid #E8EDFF !important; box-shadow: 0 2px 12px rgba(99,102,241,0.06) !important; }
.stAlert { border-radius: 12px !important; }
.stForm { background: rgba(248,250,252,0.8) !important; border: 1px solid #E8EDFF !important; border-radius: 16px !important; }
.theme-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #EEF2FF; border: 1px solid #C7D2FE;
    color: #4F46E5; font-size: 0.78rem; font-weight: 600;
    padding: 0.25rem 0.7rem; border-radius: 20px; margin-bottom: 0.5rem;
}
.predictor-result-card {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1px solid rgba(16,185,129,0.3); border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 8px 24px rgba(16,185,129,0.1);
}
.advisor-card {
    background: white; border: 1px solid #E8EDFF;
    border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.06);
}
</style>
"""

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_data():
    os.makedirs("data", exist_ok=True)
    sample_file = os.path.join("data", "student_performance_sample.csv")
    if not os.path.exists(sample_file):
        save_sample_dataset()
    return pd.read_csv(sample_file)

def section(icon: str, title: str):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # Inject theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

    # ── HEADER ──
    st.markdown("""
    <div class="app-header">
        <div class="app-header-icon">🎓</div>
        <div class="app-header-title">Student Performance Clustering System</div>
        <div class="app-header-sub">Machine Learning · Unsupervised K-Means Clustering · Academic Behavior Analytics</div>
        <div>
            <span class="app-header-badge">🤖 K-Means Clustering</span>
            &nbsp;
            <span class="app-header-badge">📊 300 Students</span>
            &nbsp;
            <span class="app-header-badge">⚡ Real-time Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──
    dark = st.session_state.dark_mode
    sb_text = "#CBD5E1" if dark else "#1E293B"

    st.sidebar.markdown(f'<div style="font-size:3rem;text-align:center;padding:0.5rem 0">🏫</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<h2 style="text-align:center;font-weight:800;font-size:1.1rem;color:{sb_text};margin-bottom:0.2rem">Controls & Setup</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Theme toggle
    theme_icon = "☀️" if dark else "🌙"
    theme_next = "Light Mode" if dark else "Dark Mode"
    badge_text = "🌑 Dark Mode" if dark else "🔆 Light Mode"
    st.sidebar.markdown(f'<div style="text-align:center"><span class="theme-badge">{badge_text}</span></div>', unsafe_allow_html=True)
    if st.sidebar.button(f"{theme_icon} Switch to {theme_next}", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<p style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#6366F1">📂 Data Source</p>', unsafe_allow_html=True)
    data_source = st.sidebar.radio(
        "Select data source:",
        ("📊 Sample Dataset (300 Students)", "📤 Upload Custom CSV File"),
        label_visibility="collapsed"
    )

    df = None
    if data_source == "📊 Sample Dataset (300 Students)":
        df = load_data()
        if st.sidebar.button("🔄 Regenerate Sample Data", use_container_width=True):
            save_sample_dataset()
            st.rerun()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Student CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} rows!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                return
        else:
            st.info("👆 Please upload a CSV file via the sidebar, or select the Sample Dataset.")
            return

    if df is None or df.empty:
        st.error("No data available.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset must have at least 2 numerical columns!")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<p style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#6366F1">🎯 Feature Selection</p>', unsafe_allow_html=True)
    default_study = next((c for c in numeric_cols if "study" in c.lower()), numeric_cols[0])
    default_attend = next((c for c in numeric_cols if "attend" in c.lower()), numeric_cols[min(1, len(numeric_cols)-1)])
    selected_features = st.sidebar.multiselect(
        "Choose features for clustering:",
        options=numeric_cols,
        default=[default_study, default_attend] if default_study != default_attend else numeric_cols[:2],
        label_visibility="collapsed"
    )

    if len(selected_features) < 2:
        st.warning("⚠️ Select at least 2 features in the sidebar to run clustering.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<p style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#6366F1">🤖 Model Parameters</p>', unsafe_allow_html=True)
    n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)
    random_state = st.sidebar.number_input("Random Seed", value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background:{'rgba(99,102,241,0.1)' if dark else '#EEF2FF'};border:1px solid {'rgba(99,102,241,0.3)' if dark else '#C7D2FE'};border-radius:12px;padding:1rem;margin-top:0.5rem">
        <div style="font-size:0.75rem;font-weight:700;color:#6366F1;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.5rem">📌 Selected Config</div>
        <div style="font-size:0.85rem;color:{'#94A3B8' if dark else '#64748B'}">Features: <b style="color:{'#A5B4FC' if dark else '#4F46E5'}">{', '.join([f.replace('_',' ') for f in selected_features])}</b></div>
        <div style="font-size:0.85rem;color:{'#94A3B8' if dark else '#64748B'}">Clusters K: <b style="color:{'#A5B4FC' if dark else '#4F46E5'}">{n_clusters}</b></div>
        <div style="font-size:0.85rem;color:{'#94A3B8' if dark else '#64748B'}">Students: <b style="color:{'#A5B4FC' if dark else '#4F46E5'}">{len(df)}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── COMPUTE ──
    X_scaled, scaler = preprocess_data(df, selected_features)
    model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    centroids_df = get_unscaled_centroids(model, scaler, selected_features)
    personas = assign_personas(centroids_df, selected_features)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Data Overview",
        "📈  Optimal K Finder",
        "🎨  Cluster Visualizer",
        "📋  Dataset & Personas",
        "🔮  Student Predictor"
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — DATA OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        # Metric cards
        section("📌", "Dataset Highlights")
        m1, m2, m3, m4 = st.columns(4)
        accent = "#818CF8" if dark else "#4F46E5"

        avg1 = df[selected_features[0]].mean()
        avg2 = df[selected_features[1]].mean()

        metrics = [
            ("👥", "Total Students", str(len(df)), "enrolled"),
            ("🔢", "Numeric Features", str(len(numeric_cols)), "available"),
            (
                "📚",
                selected_features[0].replace("_", " "),
                f"{avg1:.1f}",
                "avg hrs/week" if "study" in selected_features[0].lower() else "avg"
            ),
            (
                "🎯",
                selected_features[1].replace("_", " "),
                f"{avg2:.1f}%",
                "avg attendance" if "attend" in selected_features[1].lower() else "avg"
            ),
        ]
        for col, (icon, title, val, sub) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value" style="color:{accent}">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("📄", "Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col_stat, col_corr = st.columns([1, 1])
        with col_stat:
            section("📐", "Summary Statistics")
            st.dataframe(
                df[selected_features].describe().T[["mean", "std", "min", "50%", "max"]],
                use_container_width=True
            )
        with col_corr:
            section("🔗", "Correlation Matrix")
            corr_fig = plot_correlation_matrix(df, selected_features)
            st.plotly_chart(corr_fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — OPTIMAL K
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        section("📈", "Finding the Optimal Number of Clusters (K)")

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"""
            <div style="background:{'rgba(99,102,241,0.1)' if dark else '#EEF2FF'};border:1px solid {'rgba(99,102,241,0.25)' if dark else '#C7D2FE'};border-radius:12px;padding:1.1rem">
                <div style="font-size:1rem;font-weight:700;color:#6366F1;margin-bottom:0.4rem">📉 Elbow Method (WCSS)</div>
                <div style="font-size:0.88rem;color:{'#94A3B8' if dark else '#64748B'}">
                    Looks for the "elbow" — the point where adding more clusters gives diminishing returns in reducing within-cluster variance.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with info_col2:
            st.markdown(f"""
            <div style="background:{'rgba(16,185,129,0.1)' if dark else '#ECFDF5'};border:1px solid {'rgba(16,185,129,0.25)' if dark else '#A7F3D0'};border-radius:12px;padding:1.1rem">
                <div style="font-size:1rem;font-weight:700;color:#10B981;margin-bottom:0.4rem">🧮 Silhouette Score</div>
                <div style="font-size:0.88rem;color:{'#94A3B8' if dark else '#64748B'}">
                    Measures how well each student fits its own cluster vs. others. Range: −1 to +1. <b>Higher is better</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        eval_df = compute_elbow_and_silhouette(X_scaled, max_k=8)
        fig_eval = plot_elbow_and_silhouette(eval_df)
        st.plotly_chart(fig_eval, use_container_width=True)

        best_k = int(eval_df.loc[eval_df["Silhouette_Score"].idxmax()]["K"])
        best_score = eval_df["Silhouette_Score"].max()

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⭐</div>
                <div class="metric-title">Recommended K</div>
                <div class="metric-value" style="color:{'#818CF8' if dark else '#4F46E5'}">{best_k}</div>
                <div class="metric-sub">best silhouette</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-title">Best Sil. Score</div>
                <div class="metric-value" style="color:#10B981">{best_score:.3f}</div>
                <div class="metric-sub">at K={best_k}</div>
            </div>""", unsafe_allow_html=True)
        with rc3:
            diff = n_clusters - best_k
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎛️</div>
                <div class="metric-title">Your Selected K</div>
                <div class="metric-value" style="color:{'#FBBF24' if diff != 0 else '#10B981'}">{n_clusters}</div>
                <div class="metric-sub">{diff_str} from optimal</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — CLUSTER VISUALIZER
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        section("🎨", "K-Means Cluster Visualizer")
        x_axis = selected_features[0]
        y_axis = selected_features[1]

        fig_2d = plot_2d_clusters(df_clustered, x_axis, y_axis, "Cluster", centroids_df, personas)
        st.plotly_chart(fig_2d, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("📍", "Cluster Centroids (Original Scale)")

        centroids_display = centroids_df.copy()
        centroids_display["Persona"] = centroids_display["Cluster"].apply(
            lambda c: personas.get(c, {}).get("name", f"Cluster {c}")
        )
        cols_order = ["Cluster", "Persona"] + selected_features
        styled = (
            centroids_display[cols_order]
            .style
            .highlight_max(subset=selected_features, color="#DCFCE7")
            .highlight_min(subset=selected_features, color="#FEE2E2")
        )
        st.dataframe(styled, use_container_width=True)

        if len(selected_features) >= 3:
            st.markdown("<br>", unsafe_allow_html=True)
            section("🧊", "3D Cluster Visualizer")
            z_axis = selected_features[2]
            fig_3d = plot_3d_clusters(df_clustered, x_axis, y_axis, z_axis, "Cluster", personas)
            st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("📦", "Feature Distributions by Cluster")
        boxplots = plot_feature_boxplots(df_clustered, "Cluster", selected_features, personas)
        col_box1, col_box2 = st.columns(2)
        for i, boxfig in enumerate(boxplots):
            (col_box1 if i % 2 == 0 else col_box2).plotly_chart(boxfig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — DATASET & PERSONAS
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        section("👥", "Academic Persona Breakdown")

        counts = df_clustered["Cluster"].value_counts().to_dict()
        total_st = len(df_clustered)

        PERSONA_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]

        card_cols = st.columns(min(n_clusters, 4))
        for idx in range(n_clusters):
            p_info = personas.get(idx, {"name": f"Cluster {idx}", "description": ""})
            st_count = counts.get(idx, 0)
            pct = (st_count / total_st) * 100
            color = PERSONA_COLORS[idx % len(PERSONA_COLORS)]

            with card_cols[idx % len(card_cols)]:
                st.markdown(f"""
                <div class="persona-card" style="border-left-color:{color}">
                    <div class="persona-title">{p_info['name']}</div>
                    <div class="persona-count" style="color:{color}">{st_count}</div>
                    <div class="persona-pct">students &nbsp;·&nbsp; {pct:.1f}% of cohort</div>
                    <div class="persona-desc">{p_info['description']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("📋", "Complete Clustered Dataset")

        filter_cluster = st.multiselect(
            "Filter by Cluster:",
            options=list(range(n_clusters)),
            default=list(range(n_clusters)),
            format_func=lambda c: f"Cluster {c}: {personas.get(c, {}).get('name', '')}"
        )

        df_filtered = df_clustered[df_clustered["Cluster"].isin(filter_cluster)].copy()
        df_filtered["Persona"] = df_filtered["Cluster"].apply(lambda c: personas.get(c, {}).get("name", ""))
        st.dataframe(df_filtered, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Clustered Dataset as CSV",
            data=csv_data,
            file_name="clustered_students_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 5 — STUDENT PREDICTOR
    # ══════════════════════════════════════════════════════════════════
    with tab5:
        section("🔮", "Individual Student Cluster Predictor")

        st.markdown(f"""
        <div style="background:{'rgba(99,102,241,0.08)' if dark else '#F5F3FF'};border:1px solid {'rgba(99,102,241,0.2)' if dark else '#DDD6FE'};border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.5rem">
            <p style="margin:0;color:{'#A5B4FC' if dark else '#6D28D9'};font-size:0.95rem">
                Enter a student's academic metrics below. The trained K-Means model will instantly predict which cluster they belong to and provide tailored academic recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("predictor_form"):
            form_cols = st.columns(len(selected_features))
            user_inputs = {}

            for i, feat in enumerate(selected_features):
                mean_val = float(df[feat].mean())
                with form_cols[i]:
                    user_inputs[feat] = st.number_input(
                        f"📝 {feat.replace('_', ' ')}",
                        min_value=0.0,
                        max_value=150.0,
                        value=round(mean_val, 1),
                        step=0.5
                    )

            submit_btn = st.form_submit_button("🚀 Predict Student Cluster", use_container_width=True)

        if submit_btn:
            pred_cluster, pred_persona = predict_single_student(
                model, scaler, selected_features, user_inputs, personas
            )
            st.balloons()

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                PERSONA_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]
                pcolor = PERSONA_COLORS[pred_cluster % len(PERSONA_COLORS)]
                st.markdown(f"""
                <div class="predictor-result-card" style="border-left:5px solid {pcolor};background:{'rgba(30,41,59,0.8)' if dark else 'white'}">
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{pcolor};margin-bottom:0.5rem">
                        ✅ Assigned Cluster
                    </div>
                    <div style="font-size:2.5rem;font-weight:900;color:{pcolor};margin-bottom:0.3rem">#{pred_cluster}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{'#E2E8F0' if dark else '#1E293B'};margin-bottom:0.5rem">{pred_persona['name']}</div>
                    <div style="font-size:0.88rem;color:{'#94A3B8' if dark else '#64748B'};line-height:1.5">{pred_persona['description']}</div>
                </div>
                """, unsafe_allow_html=True)

            with p_col2:
                st.markdown(f'<div class="advisor-card">', unsafe_allow_html=True)
                st.markdown("### 💡 Recommended Advisor Action Plan")

                if "High Performers" in pred_persona["name"]:
                    st.markdown("""
                    - 🌟 **Encouragement**: Nominate for peer tutoring and advanced honors projects.
                    - 🏆 **Recognition**: Submit for academic excellence awards and scholarships.
                    - 🎯 **Growth**: Encourage research competitions, leadership roles, and mentorship programs.
                    """)
                elif "At-Risk" in pred_persona["name"]:
                    st.markdown("""
                    - ⚠️ **Immediate Action**: Schedule one-on-one academic counseling this week.
                    - 📅 **Attendance Tracking**: Enable automated reminders and mandatory check-ins.
                    - 📚 **Resources**: Connect to tutoring center, study skill workshops, and peer support.
                    """)
                elif "Passive" in pred_persona["name"]:
                    st.markdown("""
                    - 📚 **Study Habit Workshop**: Structured after-class guided practice exercises.
                    - 🤝 **Peer Support**: Pair with active study groups to boost self-study time.
                    - 🎯 **Goal Setting**: Help set measurable weekly study-hour targets.
                    """)
                elif "Independent" in pred_persona["name"]:
                    st.markdown("""
                    - ⚡ **Class Engagement**: Check for scheduling conflicts or lecture recording access.
                    - 📝 **Assignments Check**: Monitor submission consistency to prevent grade slippage.
                    - 🤝 **Connection**: Encourage virtual office-hours participation.
                    """)
                else:
                    st.markdown("- 🎯 Maintain regular progress tracking and mid-semester feedback sessions.")

                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
