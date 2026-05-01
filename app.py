import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Premium CSS — Dark professional theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0f1a0f;
    color: #e8f0e8;
}
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0a120a !important;
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4ade80 !important;
}
[data-testid="stSidebarContent"] {
    background: #0a120a;
}

/* ── HEADER BANNER ── */
.agri-header {
    background: linear-gradient(135deg, #0a2a0a 0%, #0f3d1a 50%, #0a2a0a 100%);
    border: 1px solid #1e4a1e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.agri-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(74,222,128,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.agri-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 100px;
    background: radial-gradient(ellipse, rgba(34,197,94,0.06) 0%, transparent 70%);
}
.agri-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.agri-title span { color: #4ade80; }
.agri-subtitle {
    font-size: 0.88rem;
    color: #6a9a6a;
    margin-top: 0.4rem;
    font-weight: 400;
}
.header-pills {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.hpill {
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hpill-green { background: rgba(74,222,128,0.12); border: 1px solid rgba(74,222,128,0.25); color: #4ade80; }
.hpill-blue  { background: rgba(56,189,248,0.1);  border: 1px solid rgba(56,189,248,0.2);  color: #38bdf8; }
.hpill-gold  { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.2);  color: #fbbf24; }

/* ── SECTION LABELS ── */
.sec-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a6a4a;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e3a1e;
}

/* ── STAT CARDS ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.stat-card {
    background: #111f11;
    border: 1px solid #1e3a1e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.green::before { background: linear-gradient(90deg, #16a34a, #4ade80); }
.stat-card.blue::before  { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.stat-card.gold::before  { background: linear-gradient(90deg, #b45309, #fbbf24); }
.stat-card.rose::before  { background: linear-gradient(90deg, #be185d, #fb7185); }
.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a6a4a;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.03em;
}
.stat-value.green { color: #4ade80; }
.stat-value.gold  { color: #fbbf24; }
.stat-sub {
    font-size: 0.7rem;
    color: #4a6a4a;
    margin-top: 0.2rem;
}

/* ── REGION CARD ── */
.region-info-card {
    background: linear-gradient(135deg, #111f11, #0f1f0f);
    border: 1px solid #1e4a1e;
    border-left: 3px solid #4ade80;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #8aaa8a;
}
.region-info-card b { color: #c8e6c8; }

/* ── CROP BANNER ── */
.crop-winner {
    background: linear-gradient(135deg, #052010 0%, #0a3a18 40%, #052010 100%);
    border: 1px solid #1e5a1e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.crop-winner::before {
    content: '';
    position: absolute;
    top: -50%; left: -10%;
    width: 120%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(74,222,128,0.08) 0%, transparent 60%);
}
.crop-winner-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.4rem;
}
.crop-winner-name {
    font-size: 2.4rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.04em;
    text-transform: uppercase;
}
.crop-winner-meta {
    font-size: 0.82rem;
    color: #6aaa6a;
    margin-top: 0.3rem;
}

/* ── RANK CARDS ── */
.rank-card {
    background: #111f11;
    border: 1px solid #1e3a1e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rank-card.top { border-color: #16a34a; background: linear-gradient(135deg, #0a2a12, #111f11); }
.rank-badge {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 800;
    flex-shrink: 0;
}
.rank-badge.r1 { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.rank-badge.r2 { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.rank-badge.r3 { background: rgba(180,83,9,0.12); color: #fb923c; border: 1px solid rgba(180,83,9,0.2); }
.rank-body { flex: 1; }
.rank-name { font-size: 0.95rem; font-weight: 700; color: #e8f0e8; margin-bottom: 2px; }
.rank-conf { font-size: 0.75rem; color: #4ade80; font-weight: 600; }
.rank-price { font-size: 0.72rem; color: #4a6a4a; }
.rank-bar-bg { height: 3px; background: #1e3a1e; border-radius: 2px; margin-top: 6px; }
.rank-bar-fg { height: 3px; border-radius: 2px; }

/* ── SOIL HEALTH ── */
.soil-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    background: #111f11;
    border: 1px solid #1e3a1e;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.soil-ok    { border-left: 3px solid #4ade80; }
.soil-warn  { border-left: 3px solid #fbbf24; }
.soil-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
}
.soil-badge.ok   { background: rgba(74,222,128,0.1); color: #4ade80; }
.soil-badge.warn { background: rgba(251,191,36,0.1); color: #fbbf24; }

/* ── PROFIT METRICS ── */
.profit-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.6rem;
    margin: 0.5rem 0;
}
.profit-item {
    background: #111f11;
    border: 1px solid #1e3a1e;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.profit-lbl { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #4a6a4a; margin-bottom: 0.25rem; }
.profit-val { font-size: 1.1rem; font-weight: 800; color: #4ade80; }

/* ── AI SUMMARY BOX ── */
.ai-summary {
    background: linear-gradient(135deg, #0a1a12, #0f2a1a);
    border: 1px solid #1e4a2a;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
}
.ai-summary::before {
    content: '🤖';
    position: absolute;
    top: -12px; left: 20px;
    background: #0a1a12;
    padding: 0 8px;
    font-size: 1.2rem;
}
.ai-tag {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.75rem;
}
.ai-text {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #9aba9a;
}
.ai-text b { color: #c8e6c8; }
.ai-text .hi { color: #4ade80; font-weight: 600; }

/* ── COMPARISON TABLE ── */
.compare-wrap {
    background: #0a120a;
    border: 1px solid #1e3a1e;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] {
    background: #111f11;
    border: 1px solid #1e3a1e;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
div[data-testid="stMetricValue"] { color: #4ade80 !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { color: #4a6a4a !important; font-size: 0.75rem !important; }

div[data-testid="stSelectbox"] > div > div {
    background: #111f11 !important;
    border-color: #1e3a1e !important;
    color: #e8f0e8 !important;
    border-radius: 8px !important;
}
div[data-testid="stSlider"] { color: #4a6a4a; }

.stButton > button {
    background: linear-gradient(135deg, #15803d, #16a34a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stInfo, .stSuccess, .stWarning {
    border-radius: 10px !important;
    border: none !important;
}

div[data-testid="stExpander"] {
    background: #111f11 !important;
    border: 1px solid #1e3a1e !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    color: #8aaa8a !important;
    font-size: 0.85rem !important;
}

.stDataFrame { border-radius: 10px !important; overflow: hidden; }

h1, h2, h3 { color: #c8e6c8 !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
p, li { color: #8aaa8a; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a120a; }
::-webkit-scrollbar-thumb { background: #1e3a1e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Region Data
# ─────────────────────────────────────────────
REGION_DATA = {
    "-- Select a Region --":         None,
    "🌾 Punjab (North India)":       {"N": 80, "P": 55, "K": 45, "temperature": 24.0, "humidity": 65.0, "ph": 7.8, "rainfall": 70.0,  "zone": "Semi-Arid"},
    "🌾 Haryana (North India)":      {"N": 75, "P": 50, "K": 42, "temperature": 25.0, "humidity": 60.0, "ph": 7.9, "rainfall": 65.0,  "zone": "Semi-Arid"},
    "🌿 Uttar Pradesh (Central)":    {"N": 65, "P": 45, "K": 40, "temperature": 27.0, "humidity": 70.0, "ph": 7.5, "rainfall": 100.0, "zone": "Subtropical"},
    "🌿 Madhya Pradesh (Central)":   {"N": 55, "P": 38, "K": 35, "temperature": 28.0, "humidity": 55.0, "ph": 7.2, "rainfall": 115.0, "zone": "Tropical"},
    "🌴 Maharashtra (West)":         {"N": 60, "P": 40, "K": 38, "temperature": 30.0, "humidity": 68.0, "ph": 6.8, "rainfall": 130.0, "zone": "Tropical"},
    "🌴 Gujarat (West)":             {"N": 50, "P": 35, "K": 30, "temperature": 32.0, "humidity": 55.0, "ph": 7.6, "rainfall": 80.0,  "zone": "Semi-Arid"},
    "🌿 Rajasthan (West/Desert)":    {"N": 30, "P": 20, "K": 22, "temperature": 34.0, "humidity": 35.0, "ph": 8.2, "rainfall": 40.0,  "zone": "Arid"},
    "🌿 Bihar (East)":               {"N": 70, "P": 48, "K": 42, "temperature": 27.0, "humidity": 75.0, "ph": 7.0, "rainfall": 120.0, "zone": "Subtropical"},
    "🌿 West Bengal (East)":         {"N": 78, "P": 52, "K": 46, "temperature": 28.0, "humidity": 80.0, "ph": 6.5, "rainfall": 175.0, "zone": "Humid"},
    "🌿 Odisha (East)":              {"N": 65, "P": 44, "K": 40, "temperature": 29.0, "humidity": 78.0, "ph": 6.4, "rainfall": 155.0, "zone": "Tropical"},
    "🌴 Andhra Pradesh (South)":     {"N": 62, "P": 42, "K": 38, "temperature": 31.0, "humidity": 72.0, "ph": 6.6, "rainfall": 120.0, "zone": "Tropical"},
    "🌴 Telangana (South)":          {"N": 58, "P": 38, "K": 35, "temperature": 32.0, "humidity": 68.0, "ph": 6.7, "rainfall": 105.0, "zone": "Tropical"},
    "🌴 Tamil Nadu (South)":         {"N": 60, "P": 40, "K": 36, "temperature": 33.0, "humidity": 74.0, "ph": 6.5, "rainfall": 100.0, "zone": "Tropical"},
    "🌴 Karnataka (South)":          {"N": 55, "P": 36, "K": 34, "temperature": 29.0, "humidity": 70.0, "ph": 6.8, "rainfall": 130.0, "zone": "Tropical"},
    "🌴 Kerala (South)":             {"N": 70, "P": 50, "K": 55, "temperature": 28.0, "humidity": 85.0, "ph": 5.8, "rainfall": 280.0, "zone": "Humid Tropical"},
    "🏔️ Himachal Pradesh (Hills)":   {"N": 45, "P": 30, "K": 28, "temperature": 15.0, "humidity": 70.0, "ph": 6.2, "rainfall": 150.0, "zone": "Temperate"},
    "🏔️ Uttarakhand (Hills)":        {"N": 48, "P": 32, "K": 30, "temperature": 17.0, "humidity": 72.0, "ph": 6.3, "rainfall": 160.0, "zone": "Temperate"},
    "🌿 Jharkhand (East)":           {"N": 52, "P": 35, "K": 33, "temperature": 27.0, "humidity": 73.0, "ph": 5.9, "rainfall": 140.0, "zone": "Tropical"},
    "🌿 Chhattisgarh (Central)":     {"N": 58, "P": 38, "K": 36, "temperature": 29.0, "humidity": 70.0, "ph": 6.1, "rainfall": 135.0, "zone": "Tropical"},
    "🌿 Assam (Northeast)":          {"N": 72, "P": 48, "K": 44, "temperature": 26.0, "humidity": 82.0, "ph": 5.7, "rainfall": 250.0, "zone": "Humid"},
    "✏️ Custom (Enter Manually)":    "custom",
}

MARKET_PRICE = {
    "rice": 20, "wheat": 18, "maize": 22, "cotton": 60,
    "sugarcane": 5, "banana": 25, "mango": 50,
    "apple": 80, "grapes": 70, "watermelon": 10,
    "muskmelon": 15, "papaya": 18, "coconut": 30,
    "jute": 12, "coffee": 90, "chickpea": 40,
    "kidneybeans": 35, "pigeonpeas": 30, "mothbeans": 28,
    "mungbean": 30, "blackgram": 28, "lentil": 32,
    "pomegranate": 55, "orange": 35
}

# ─────────────────────────────────────────────
# Load & Train
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

@st.cache_resource
def train_model(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    train_acc   = clf.score(X_train, y_train)
    test_acc    = clf.score(X_test, y_test)
    y_pred      = clf.predict(X_test)
    report      = classification_report(y_test, y_pred, output_dict=True)
    importances = pd.Series(clf.feature_importances_, index=_X.columns).sort_values(ascending=False)
    return clf, train_acc, test_acc, report, importances

try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ `Crop_recommendation.csv` not found. Place it in the same folder as `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Dataset Error: {e}")
    st.stop()

required_cols = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}
if not required_cols.issubset(data.columns):
    st.error(f"❌ Missing columns. Found: {list(data.columns)}")
    st.stop()

X = data.drop("label", axis=1)
y = data["label"]
model, train_acc, test_acc, report, feature_importances = train_model(X, y)

# ─────────────────────────────────────────────
# Chart helper — dark theme
# ─────────────────────────────────────────────
BG    = "#0f1a0f"
BG2   = "#111f11"
GRID  = "#1e3a1e"
GREEN = "#4ade80"
GREENS = ["#4ade80","#22c55e","#16a34a","#86efac","#bbf7d0"]
MULTI  = ["#4ade80","#38bdf8","#fbbf24","#fb923c","#a78bfa","#fb7185","#34d399"]

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors="#4a6a4a", labelsize=8)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    return fig, ax

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.5rem 0 1rem 0'>
      <div style='font-size:1.5rem'>🌾</div>
      <div style='font-size:1rem;font-weight:800;color:#e8f0e8;letter-spacing:-0.02em'>AgriSense AI</div>
      <div style='font-size:0.68rem;color:#4a6a4a;letter-spacing:0.08em;text-transform:uppercase'>Smart Crop Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec-label">Model Performance</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Train Acc", f"{round(train_acc*100,1)}%",
                  help="Score on training data")
    with col_b:
        st.metric("Test Acc ✅", f"{round(test_acc*100,1)}%",
                  help="Score on unseen 20% hold-out — the real accuracy")

    if train_acc == 1.0:
        st.caption("ℹ️ Train = 100% is normal for Random Forest. Test Accuracy is what counts.")

    st.divider()
    st.markdown('<div class="sec-label">Dataset Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Samples", data.shape[0])
    with c2: st.metric("Crops",   y.nunique())
    with c3: st.metric("Features", X.shape[1])

    st.divider()
    st.markdown('<div class="sec-label">Crop Distribution</div>', unsafe_allow_html=True)
    crop_counts = y.value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]
    st.dataframe(crop_counts, use_container_width=True, hide_index=True, height=220)

    st.divider()
    st.caption("80/20 Train-Test Split · RandomForest n=100")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="agri-header">
  <div class="agri-title">🌱 Agri<span>Sense</span> AI</div>
  <div class="agri-subtitle">Smart Agriculture Decision System — Powered by Random Forest & Real Indian Agro-Climate Data</div>
  <div class="header-pills">
    <span class="hpill hpill-green">Random Forest ML</span>
    <span class="hpill hpill-blue">22+ Crops</span>
    <span class="hpill hpill-gold">20 Indian Regions</span>
    <span class="hpill hpill-green">Profit Analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP 1 — Region
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">📍 Step 1 — Select Your Region</div>', unsafe_allow_html=True)

selected_region = st.selectbox(
    "Choose an Indian state / region:",
    options=list(REGION_DATA.keys()),
    index=0,
    label_visibility="collapsed"
)

region_info = REGION_DATA[selected_region]

if region_info is None:
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;color:#4a6a4a'>
      <div style='font-size:3rem;opacity:0.3'>🗺️</div>
      <div style='font-size:1rem;font-weight:600;color:#6a8a6a;margin-top:0.5rem'>Select a region above to begin</div>
      <div style='font-size:0.82rem;margin-top:0.3rem'>Soil & climate data will auto-fill for your chosen state</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

is_custom = (region_info == "custom")

# ─────────────────────────────────────────────
# STEP 2 — Inputs
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">🧪 Step 2 — Soil & Climate Parameters</div>', unsafe_allow_html=True)

if not is_custom:
    zone = region_info['zone']
    st.markdown(f"""
    <div class="region-info-card">
      📌 <b>{selected_region}</b> &nbsp;·&nbsp; Climate Zone: <b>{zone}</b><br>
      <span style='font-size:0.78rem'>Values auto-filled from average regional soil & climate data — fine-tune with sliders if needed.</span>
    </div>
    """, unsafe_allow_html=True)
    defaults = region_info
else:
    st.info("✏️ Custom mode — enter your own values below.")
    defaults = {"N": 50, "P": 50, "K": 50, "temperature": 25.0,
                "humidity": 70.0, "ph": 6.5, "rainfall": 100.0, "zone": "Custom"}

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌿 Soil Nutrients (kg/ha)**")
    N        = st.slider("Nitrogen (N)",    0,   140, int(defaults["N"]),            help="Nitrogen content in soil")
    P        = st.slider("Phosphorus (P)",  5,   145, int(defaults["P"]),            help="Phosphorus content in soil")
    K        = st.slider("Potassium (K)",   5,   205, int(defaults["K"]),            help="Potassium content in soil")

with col2:
    st.markdown("**🌡️ Climate Conditions**")
    temp     = st.slider("Temperature (°C)", 8.0,  44.0, float(defaults["temperature"]), step=0.5)
    humidity = st.slider("Humidity (%)",    14.0, 100.0, float(defaults["humidity"]),    step=0.5)
    rainfall = st.slider("Rainfall (mm)",   20.0, 300.0, float(defaults["rainfall"]),    step=1.0)

with col3:
    st.markdown("**🧂 Soil pH**")
    ph = st.slider("pH Value", 3.5, 10.0, float(defaults["ph"]), step=0.1,
                   help="Soil acidity/alkalinity — ideal range: 5.5–7.5")
    # pH visual gauge
    ph_pct = (ph - 3.5) / (10.0 - 3.5) * 100
    ph_color = "#4ade80" if 5.5 <= ph <= 7.5 else "#fbbf24" if ph < 5.5 else "#fb923c"
    ph_label = "Optimal ✅" if 5.5 <= ph <= 7.5 else ("Acidic ⚠️" if ph < 5.5 else "Alkaline ⚠️")
    st.markdown(f"""
    <div style='margin-top:0.5rem'>
      <div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#4a6a4a;margin-bottom:4px'>
        <span>Acidic (3.5)</span><span>Neutral</span><span>Alkaline (10)</span>
      </div>
      <div style='height:6px;background:#1e3a1e;border-radius:3px;overflow:hidden'>
        <div style='height:6px;width:{ph_pct:.1f}%;background:{ph_color};border-radius:3px;transition:width 0.3s'></div>
      </div>
      <div style='font-size:0.78rem;color:{ph_color};font-weight:600;margin-top:5px'>{ph:.1f} — {ph_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP 3 — Predict Button
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">🚀 Step 3 — Get AI Recommendation</div>', unsafe_allow_html=True)

if st.button("🌾 Analyse & Recommend Crop for This Region"):
    input_data = pd.DataFrame(
        [[N, P, K, temp, humidity, ph, rainfall]],
        columns=X.columns
    )

    try:
        prediction = model.predict(input_data)[0]
        probs      = model.predict_proba(input_data)[0]
        classes    = model.classes_
        top3_idx   = np.argsort(probs)[-3:][::-1]
        top5_idx   = np.argsort(probs)[-5:][::-1]
        region_label = selected_region if not is_custom else "Custom Region"
        conf_pct   = round(probs[top3_idx[0]] * 100, 1)

        # ── Profit calc
        yield_per_ha = (rainfall * 0.008) + (temp * 0.05) + (N * 0.02)
        price        = MARKET_PRICE.get(prediction.lower(), 15)
        profit       = yield_per_ha * price * 1000

        # ── WINNER BANNER
        st.markdown(f"""
        <div class="crop-winner">
          <div class="crop-winner-label">🏆 Best Crop Recommendation for {region_label}</div>
          <div class="crop-winner-name">🌾 {prediction.upper()}</div>
          <div class="crop-winner-meta">{conf_pct}% confidence &nbsp;·&nbsp; ₹{price}/kg market price &nbsp;·&nbsp; Est. profit ₹{round(profit):,}/ha</div>
        </div>
        """, unsafe_allow_html=True)

        # ── STAT ROW
        zone_label = region_info.get("zone", "Custom") if not is_custom else "Custom"
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card green">
            <div class="stat-label">Confidence</div>
            <div class="stat-value green">{conf_pct}%</div>
            <div class="stat-sub">Model certainty</div>
          </div>
          <div class="stat-card gold">
            <div class="stat-label">Market Price</div>
            <div class="stat-value gold">₹{price}</div>
            <div class="stat-sub">Per kilogram</div>
          </div>
          <div class="stat-card blue">
            <div class="stat-label">Est. Profit</div>
            <div class="stat-value" style="color:#38bdf8;font-size:1.3rem">₹{round(profit/1000,1)}k</div>
            <div class="stat-sub">Per hectare</div>
          </div>
          <div class="stat-card rose">
            <div class="stat-label">Climate Zone</div>
            <div class="stat-value" style="color:#fb7185;font-size:1rem">{zone_label}</div>
            <div class="stat-sub">{region_label.split('(')[0].strip() if not is_custom else 'Custom'}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TWO COLUMN LAYOUT
        left, right = st.columns([1, 1.5], gap="large")

        with left:
            # Top 3 Crops
            st.markdown('<div class="sec-label">🏆 Top 3 Crop Recommendations</div>', unsafe_allow_html=True)
            medals     = ["🥇", "🥈", "🥉"]
            rank_cls   = ["r1", "r2", "r3"]
            top_colors = ["#fbbf24", "#94a3b8", "#fb923c"]
            for rank, i in enumerate(top3_idx):
                cname = classes[i].capitalize()
                conf  = round(probs[i] * 100, 1)
                cprice = MARKET_PRICE.get(classes[i].lower(), 15)
                is_top = rank == 0
                bar_color = top_colors[rank]
                st.markdown(f"""
                <div class="rank-card {'top' if is_top else ''}">
                  <div class="rank-badge {rank_cls[rank]}">{medals[rank]}</div>
                  <div class="rank-body">
                    <div class="rank-name">{cname}</div>
                    <div class="rank-conf">{conf}% confidence</div>
                    <div class="rank-price">₹{cprice}/kg market price</div>
                    <div class="rank-bar-bg">
                      <div class="rank-bar-fg" style="width:{conf}%;background:{bar_color}"></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Profit Breakdown
            st.markdown('<div class="sec-label">💰 Profit Estimation</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="profit-grid">
              <div class="profit-item">
                <div class="profit-lbl">Yield</div>
                <div class="profit-val">{round(yield_per_ha*1000,0):.0f}</div>
                <div style='font-size:0.65rem;color:#4a6a4a'>kg/ha</div>
              </div>
              <div class="profit-item">
                <div class="profit-lbl">Price</div>
                <div class="profit-val">₹{price}</div>
                <div style='font-size:0.65rem;color:#4a6a4a'>per kg</div>
              </div>
              <div class="profit-item">
                <div class="profit-lbl">Profit</div>
                <div class="profit-val">₹{round(profit/1000,1)}k</div>
                <div style='font-size:0.65rem;color:#4a6a4a'>per ha</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Soil Health
            st.markdown('<div class="sec-label">🌱 Soil Health Analysis</div>', unsafe_allow_html=True)

            def soil_row(label, ok, ok_msg, warn_msg):
                cls = "soil-ok" if ok else "soil-warn"
                badge_cls = "ok" if ok else "warn"
                badge_txt = "Good" if ok else "Action needed"
                msg = ok_msg if ok else warn_msg
                st.markdown(f"""
                <div class="soil-row {cls}">
                  <span style="font-size:0.82rem;color:#8aaa8a">{label}: {msg}</span>
                  <span class="soil-badge {badge_cls}">{badge_txt}</span>
                </div>
                """, unsafe_allow_html=True)

            soil_row("pH",       5.5 <= ph <= 7.5,  f"{ph:.1f} — Optimal", f"{ph:.1f} — {'Add lime ↑' if ph<5.5 else 'Add sulfur ↓'}")
            soil_row("Nitrogen", N >= 20,             f"N={N} kg/ha",         f"N={N} — Apply urea/compost")
            soil_row("Phosphorus", P >= 10,           f"P={P} kg/ha",         f"P={P} — Apply superphosphate")
            soil_row("Potassium", K >= 10,            f"K={K} kg/ha",         f"K={K} — Apply KCl")
            soil_row("Rainfall", rainfall >= 50,      f"{rainfall} mm — Adequate", f"{rainfall} mm — Consider irrigation")
            soil_row("Temperature", 15 <= temp <= 38, f"{temp}°C — Suitable", f"{temp}°C — Stress conditions")

        with right:
            # Parameter Bar Chart
            st.markdown('<div class="sec-label">📊 Region Parameter Overview</div>', unsafe_allow_html=True)
            labels = ["N", "P", "K", "Temp", "Humidity", "pH×10", "Rain/10"]
            values = [N, P, K, temp, humidity, ph*10, rainfall/10]
            raw    = [N, P, K, temp, humidity, ph, rainfall]
            raw_labels = ["N","P","K","Temp°C","Humidity%","pH","Rainfall mm"]

            fig1, ax1 = dark_fig(6, 3.2)
            bars = ax1.bar(raw_labels, raw, color=MULTI[:7], edgecolor=BG, linewidth=0.8, width=0.6, zorder=3)
            for bar, val in zip(bars, raw):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         str(round(val,1)), ha="center", va="bottom", fontsize=7.5, color="#6aaa6a")
            ax1.set_title(f"Soil & Climate — {region_label.split('(')[0].strip()}", fontsize=9, fontweight="bold", color="#8aaa8a", pad=8)
            ax1.tick_params(axis='x', rotation=15)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            # Feature Importance
            st.markdown('<div class="sec-label">🔍 Feature Importance</div>', unsafe_allow_html=True)
            fig3, ax3 = dark_fig(6, 2.8)
            fi_rev = feature_importances[::-1]
            fi_colors = [GREEN if v == feature_importances.max() else "#22c55e" if v >= feature_importances.median() else "#166534"
                         for v in fi_rev.values]
            bars3 = ax3.barh(fi_rev.index, fi_rev.values, color=fi_colors, edgecolor=BG, linewidth=0.5, height=0.6, zorder=3)
            for bar, val in zip(bars3, fi_rev.values):
                ax3.text(val+0.002, bar.get_y()+bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=7.5, color="#6aaa6a")
            ax3.set_xlabel("Importance Score", fontsize=8, color="#4a6a4a")
            ax3.set_title("Which factors drive the prediction?", fontsize=9, fontweight="bold", color="#8aaa8a", pad=8)
            ax3.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
            ax3.grid(axis="y", visible=False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # Top 5 Probability
            st.markdown('<div class="sec-label">📈 Crop Probability Breakdown</div>', unsafe_allow_html=True)
            top5_labels = [classes[i].capitalize() for i in top5_idx]
            top5_probs  = [probs[i] * 100 for i in top5_idx]

            fig2, ax2 = dark_fig(6, 2.8)
            colors2 = [GREEN if i == 0 else GREENS[min(i,4)] for i in range(len(top5_labels))]
            bars2 = ax2.barh(top5_labels[::-1], top5_probs[::-1],
                             color=colors2[::-1], edgecolor=BG, linewidth=0.5, height=0.55, zorder=3)
            for i, val in enumerate(top5_probs[::-1]):
                ax2.text(val+0.4, i, f"{round(val,1)}%", va="center", fontsize=7.5, color="#6aaa6a")
            ax2.set_xlabel("Confidence (%)", fontsize=8, color="#4a6a4a")
            ax2.set_title("Top 5 Crop Probabilities", fontsize=9, fontweight="bold", color="#8aaa8a", pad=8)
            ax2.set_xlim(0, max(top5_probs)*1.15)
            ax2.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
            ax2.grid(axis="y", visible=False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── Multi-Region Comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">🗺️ Multi-Region Crop Comparison (All India)</div>', unsafe_allow_html=True)
        st.caption("AI predictions across all 20 Indian regions — your selected region is highlighted.")

        compare_rows = []
        for rname, rdata in REGION_DATA.items():
            if rdata is None or rdata == "custom":
                continue
            r_input  = pd.DataFrame([[rdata["N"], rdata["P"], rdata["K"],
                                      rdata["temperature"], rdata["humidity"],
                                      rdata["ph"], rdata["rainfall"]]], columns=X.columns)
            r_pred   = model.predict(r_input)[0]
            r_probs  = model.predict_proba(r_input)[0]
            r_conf   = round(max(r_probs) * 100, 1)
            r_price  = MARKET_PRICE.get(r_pred.lower(), 15)
            r_yield  = (rdata["rainfall"]*0.008) + (rdata["temperature"]*0.05) + (rdata["N"]*0.02)
            r_profit = round(r_yield * r_price * 1000, 0)
            clean_name = rname
            for p in ["🌾 ","🌿 ","🌴 ","🏔️ "]:
                clean_name = clean_name.replace(p, "")
            compare_rows.append({
                "Region":              clean_name,
                "Zone":                rdata["zone"],
                "Recommended Crop":    r_pred.capitalize(),
                "Confidence (%)":      r_conf,
                "Market Price (₹/kg)": r_price,
                "Est. Profit (₹/ha)":  f"₹{int(r_profit):,}"
            })

        compare_df = pd.DataFrame(compare_rows)

        def highlight_selected(row):
            clean = selected_region
            for p in ["🌾 ","🌿 ","🌴 ","🏔️ "]:
                clean = clean.replace(p, "")
            return (["background-color: #0a2a12; color: #4ade80"] * len(row)
                    if row["Region"] in clean else [""] * len(row))

        st.dataframe(
            compare_df.style.apply(highlight_selected, axis=1),
            use_container_width=True, hide_index=True, height=400
        )

        # ── AI Summary
        st.markdown("<br>", unsafe_allow_html=True)
        nutrient_status     = "adequate" if N >= 20 and P >= 10 and K >= 10 else "partially deficient"
        climate_suitability = "favorable" if 20 <= temp <= 35 and humidity >= 50 else "moderate"
        top2_name  = classes[top3_idx[1]].capitalize() if len(top3_idx) > 1 else "—"
        top2_conf  = round(probs[top3_idx[1]] * 100, 1) if len(top3_idx) > 1 else 0

        st.markdown(f"""
        <div class="ai-summary">
          <div class="ai-tag">🤖 AI Recommendation Summary</div>
          <div class="ai-text">
            The Random Forest model analysed <b>{len(classes)} crop varieties</b> using 7 soil & climate features
            and identified <span class="hi">{prediction.capitalize()}</span> as the optimal crop for
            <b>{region_label}</b> with <span class="hi">{conf_pct}% confidence</span>.<br><br>
            <b>Key findings:</b><br>
            🌡️ Climate conditions are <b>{climate_suitability}</b> — temperature {temp}°C, humidity {humidity}%, rainfall {rainfall} mm.<br>
            🌿 Soil nutrient levels are <b>{nutrient_status}</b> (N={N}, P={P}, K={K} kg/ha).<br>
            🧂 Soil pH is <b>{"optimal (5.5–7.5)" if 5.5<=ph<=7.5 else f"{'acidic' if ph<5.5 else 'alkaline'} at {ph} — amendment recommended"}</b>.<br>
            💰 Estimated profit of <span class="hi">₹{round(profit):,}/ha</span> based on current market price of ₹{price}/kg.<br>
            🥈 Runner-up: <b>{top2_name}</b> at {top2_conf}% confidence — viable alternative if market prices shift.<br><br>
            <span style='font-size:0.78rem;color:#4a6a4a'>⚠ Validate with actual soil test reports from your local agricultural office before production decisions.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# ─────────────────────────────────────────────
# Expanders
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ Debug Info & Dataset Preview"):
    st.write("**Dataset Shape:**", data.shape)
    st.write("**Columns:**", list(data.columns))
    st.dataframe(data.head(8), use_container_width=True)

with st.expander("📋 Full Model Classification Report (Test Set)"):
    st.caption("Precision, Recall, F1-Score per crop — evaluated on 20% unseen test data")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)
    st.caption(
        f"✅ Test Accuracy: **{round(test_acc*100,2)}%** | "
        f"Train Accuracy: **{round(train_acc*100,2)}%** | "
        f"Split: 80% train / 20% test | Algorithm: RandomForest(n_estimators=100)"
    )
