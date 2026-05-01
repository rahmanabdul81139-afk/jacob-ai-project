import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f4f9f4; }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1b5e20; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }
    .crop-badge {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .region-card {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 4px solid #2e7d32;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    h1 { color: #1b5e20; }
    h2, h3 { color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Region Data — Indian States
# N, P, K, temperature, humidity, ph, rainfall
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
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )
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
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Overview")
    st.metric("Train Accuracy", f"{round(train_acc * 100, 2)}%",
              help="Score on training data — expected to be high")
    st.metric("Test Accuracy ✅", f"{round(test_acc * 100, 2)}%",
              help="Score on unseen 20% hold-out data — this is the REAL accuracy")
    st.caption("ℹ️ 80/20 Train-Test Split | RandomForest(n=100, random_state=42)")
    if train_acc == 1.0:
        st.warning("Train = 100% is normal for Random Forest. **Test Accuracy** is what counts.")
    st.divider()
    st.metric("Total Samples",  data.shape[0])
    st.metric("Crops Covered",  y.nunique())
    st.metric("Features Used",  X.shape[1])
    st.divider()
    st.subheader("📁 Dataset Preview")
    st.dataframe(data.head(5), use_container_width=True)
    st.divider()
    st.subheader("🌾 Crops in Dataset")
    crop_counts = y.value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]
    st.dataframe(crop_counts, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🌱 AI Smart Agriculture Decision System")
st.caption("Select a region to auto-fill soil & climate data, then get AI-powered crop recommendations.")
st.divider()

# ─────────────────────────────────────────────
# STEP 1 — Region Selection
# ─────────────────────────────────────────────
st.subheader("📍 Step 1: Select Your Region")
selected_region = st.selectbox(
    "Choose an Indian state / region:",
    options=list(REGION_DATA.keys()),
    index=0
)

region_info = REGION_DATA[selected_region]

if region_info is None:
    st.info("👆 Select a region above to auto-fill soil & climate data, or choose **Custom** to enter manually.")
    st.stop()

is_custom = (region_info == "custom")

# ─────────────────────────────────────────────
# STEP 2 — Soil & Climate Inputs
# ─────────────────────────────────────────────
st.subheader("🧪 Step 2: Soil & Environmental Data")

if not is_custom:
    st.markdown(f"""
    <div class="region-card">
        📌 <b>{selected_region}</b> — Climate Zone: <b>{region_info['zone']}</b><br>
        Values auto-filled from average regional soil & climate data. Fine-tune with the sliders if needed.
    </div>
    """, unsafe_allow_html=True)
    defaults = region_info
else:
    st.info("✏️ Custom mode — enter your own values below.")
    defaults = {"N": 50, "P": 50, "K": 50, "temperature": 25.0,
                "humidity": 70.0, "ph": 6.5, "rainfall": 100.0, "zone": "Custom"}

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌿 Soil Nutrients**")
    N        = st.slider("Nitrogen (N)",     0,    140,  int(defaults["N"]),            help="Nitrogen content in soil (kg/ha)")
    P        = st.slider("Phosphorus (P)",   5,    145,  int(defaults["P"]),            help="Phosphorus content in soil (kg/ha)")
    K        = st.slider("Potassium (K)",    5,    205,  int(defaults["K"]),            help="Potassium content in soil (kg/ha)")

with col2:
    st.markdown("**🌡️ Climate Conditions**")
    temp     = st.slider("Temperature (°C)", 8.0,  44.0, float(defaults["temperature"]), step=0.5)
    humidity = st.slider("Humidity (%)",    14.0, 100.0, float(defaults["humidity"]),    step=0.5)
    rainfall = st.slider("Rainfall (mm)",   20.0, 300.0, float(defaults["rainfall"]),    step=1.0)

with col3:
    st.markdown("**🧂 Soil pH**")
    ph = st.slider("pH Value", 3.5, 10.0, float(defaults["ph"]), step=0.1,
                   help="Soil acidity/alkalinity (ideal: 5.5–7.5)")
    st.markdown("**pH Guide:**")
    st.markdown("🔴 < 5.5 → Acidic")
    st.markdown("🟢 5.5–7.5 → Optimal")
    st.markdown("🟠 > 7.5 → Alkaline")

st.divider()

# ─────────────────────────────────────────────
# STEP 3 — Predict
# ─────────────────────────────────────────────
st.subheader("🚀 Step 3: Get AI Recommendation")

if st.button("🌾 Analyze & Recommend Crop for This Region"):
    input_data = pd.DataFrame(
        [[N, P, K, temp, humidity, ph, rainfall]],
        columns=X.columns
    )

    try:
        prediction = model.predict(input_data)[0]
        probs      = model.predict_proba(input_data)[0]
        classes    = model.classes_
        top3_idx   = np.argsort(probs)[-3:][::-1]
        region_label = selected_region if not is_custom else "Custom Region"

        # Banner
        st.markdown(
            f'<div class="crop-badge">🌾 Best Crop for {region_label}: {prediction.upper()}</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1, 1.5])

        with left:
            st.subheader("🏆 Top 3 Crop Recommendations")
            medals = ["🥇", "🥈", "🥉"]
            for rank, i in enumerate(top3_idx):
                conf  = round(probs[i] * 100, 2)
                price = MARKET_PRICE.get(classes[i].lower(), 15)
                st.markdown(f"""
                <div class="metric-card">
                    {medals[rank]} <b>{classes[i].capitalize()}</b><br>
                    <span style="color:#2e7d32;font-size:1.1rem">{conf}% confidence</span><br>
                    <span style="color:#555;font-size:0.85rem">₹{price}/kg market price</span>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("💰 Profit Estimation")
            yield_per_ha = (rainfall * 0.008) + (temp * 0.05) + (N * 0.02)
            price        = MARKET_PRICE.get(prediction.lower(), 15)
            profit       = yield_per_ha * price * 1000
            st.metric("Estimated Yield (kg/ha)",  f"{round(yield_per_ha * 1000, 1)}")
            st.metric("Market Price (₹/kg)",       f"₹ {price}")
            st.metric("Estimated Profit (₹/ha)",   f"₹ {round(profit, 2):,}")

            st.subheader("🌱 Soil Health Insight")
            if ph < 5.5:
                st.warning("⚠️ Acidic soil. Add agricultural lime to raise pH.")
            elif ph > 7.5:
                st.warning("⚠️ Alkaline soil. Add sulfur or organic compost to lower pH.")
            else:
                st.success("✅ Soil pH is optimal (5.5–7.5).")
            if N < 20: st.warning("⚠️ Low Nitrogen — apply urea or compost.")
            if P < 10: st.warning("⚠️ Low Phosphorus — apply superphosphate.")
            if K < 10: st.warning("⚠️ Low Potassium — apply potassium chloride.")

        with right:
            # Input bar chart
            st.subheader("📊 Region Parameter Overview")
            labels = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
            values = [N, P, K, temp, humidity, ph, rainfall]
            colors = ["#4caf50","#66bb6a","#a5d6a7","#1976d2","#42a5f5","#ff7043","#26c6da"]
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            bars = ax1.bar(labels, values, color=colors, edgecolor="white")
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(val), ha="center", va="bottom", fontsize=8)
            ax1.set_facecolor("#f4f9f4"); fig1.patch.set_facecolor("#f4f9f4")
            ax1.set_title(f"Inputs — {region_label}", fontsize=10, fontweight="bold")
            plt.xticks(rotation=15, fontsize=8); plt.tight_layout()
            st.pyplot(fig1)

            # Feature importance
            st.subheader("🔍 Feature Importance")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            fi_colors = ["#2e7d32" if v == feature_importances.max() else "#81c784"
                         for v in feature_importances.values]
            ax3.barh(feature_importances.index[::-1], feature_importances.values[::-1],
                     color=fi_colors[::-1], edgecolor="white")
            ax3.set_xlabel("Importance Score", fontsize=9)
            ax3.set_title("Which factors drive the prediction?", fontsize=10, fontweight="bold")
            ax3.set_facecolor("#f4f9f4"); fig3.patch.set_facecolor("#f4f9f4")
            plt.tight_layout(); st.pyplot(fig3)

            # Top 5 probability
            st.subheader("📈 Crop Probability Breakdown")
            top5_idx    = np.argsort(probs)[-5:][::-1]
            top5_labels = [classes[i].capitalize() for i in top5_idx]
            top5_probs  = [probs[i] * 100 for i in top5_idx]
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            colors2 = ["#2e7d32" if i == 0 else "#81c784" for i in range(len(top5_labels))]
            ax2.barh(top5_labels[::-1], top5_probs[::-1], color=colors2[::-1], edgecolor="white")
            for i, val in enumerate(top5_probs[::-1]):
                ax2.text(val + 0.3, i, f"{round(val,1)}%", va="center", fontsize=8)
            ax2.set_xlabel("Confidence (%)", fontsize=9)
            ax2.set_title("Top 5 Crop Probabilities", fontsize=10, fontweight="bold")
            ax2.set_facecolor("#f4f9f4"); fig2.patch.set_facecolor("#f4f9f4")
            plt.tight_layout(); st.pyplot(fig2)

        # ── Multi-Region Comparison Table
        st.divider()
        st.subheader("🗺️ Multi-Region Crop Comparison")
        st.caption("AI predictions for all regions — see how crop recommendations vary across India.")

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
                "Region":               clean_name,
                "Zone":                 rdata["zone"],
                "Recommended Crop":     r_pred.capitalize(),
                "Confidence (%)":       r_conf,
                "Market Price (₹/kg)":  r_price,
                "Est. Profit (₹/ha)":   f"₹{int(r_profit):,}"
            })

        compare_df = pd.DataFrame(compare_rows)

        def highlight_selected(row):
            clean = selected_region
            for p in ["🌾 ","🌿 ","🌴 ","🏔️ "]:
                clean = clean.replace(p, "")
            return (["background-color: #c8e6c9"] * len(row)
                    if row["Region"] in clean else [""] * len(row))

        st.dataframe(
            compare_df.style.apply(highlight_selected, axis=1),
            use_container_width=True, hide_index=True
        )

        # ── AI Summary
        st.divider()
        st.subheader("🤖 AI Recommendation Summary")
        nutrient_status     = "adequate" if N >= 20 and P >= 10 and K >= 10 else "partially deficient"
        climate_suitability = "favorable" if 20 <= temp <= 35 and humidity >= 50 else "moderate"
        zone_label          = region_info.get("zone", "Custom") if not is_custom else "Custom"

        st.info(f"""
**Region:** {region_label}  |  **Climate Zone:** {zone_label}
**Recommended Crop:** {prediction.capitalize()}

The AI model analyzed **{len(classes)} crop types** and identified **{prediction.capitalize()}**
as the most profitable crop for **{region_label}** with **{round(probs[top3_idx[0]] * 100, 1)}% confidence**.

- 🌡️ Climate conditions are **{climate_suitability}** for crop growth.
- 🌿 Soil nutrients are **{nutrient_status}**.
- 💧 Rainfall: **{rainfall} mm** | Temperature: **{temp}°C**
- 💰 Estimated profit: **₹{round(profit, 2):,}/ha** under current regional conditions.

> Validate with actual soil test reports from your local agricultural office for production use.
        """)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# ─────────────────────────────────────────────
# Expanders
# ─────────────────────────────────────────────
with st.expander("⚙️ Debug Info"):
    st.write("**Dataset Shape:**", data.shape)
    st.write("**Columns:**", list(data.columns))
    st.dataframe(y.value_counts().reset_index().rename(columns={"index": "Crop", "label": "Count"}))

with st.expander("📋 Full Model Classification Report (Test Set)"):
    st.caption("Precision, Recall, F1-Score per crop — evaluated on 20% unseen test data")
    report_df = pd.DataFrame(report).T.round(2)
    st.dataframe(report_df, use_container_width=True)
    st.caption(
        f"✅ Test Accuracy: **{round(test_acc*100,2)}%** | "
        f"Train Accuracy: **{round(train_acc*100,2)}%** | "
        f"Split: 80% train / 20% test"
    )
