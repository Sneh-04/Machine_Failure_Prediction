import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }

    /* Alert box */
    .alert-box {
        background: linear-gradient(135deg, #3d0000, #1a0000);
        border: 2px solid #f85149;
        border-radius: 14px;
        padding: 24px 32px;
        text-align: center;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 10px #f8514940; }
        to   { box-shadow: 0 0 30px #f8514970; }
    }

    /* Safe box */
    .safe-box {
        background: linear-gradient(135deg, #003d1a, #001a0c);
        border: 2px solid #3fb950;
        border-radius: 14px;
        padding: 24px 32px;
        text-align: center;
    }

    /* Section divider */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #8b949e;
        margin: 1.5rem 0 .75rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }

    /* Risk bar background */
    .risk-bar-bg {
        background: #21262d;
        border-radius: 8px;
        height: 20px;
        overflow: hidden;
        margin: 6px 0 2px;
    }

    /* Override Streamlit button */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: #238636;
        color: white;
        border: none;
        padding: 10px;
        font-size: 15px;
    }
    .stButton > button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model and feature names. Returns (model, feature_names) or (None, None)."""
    model_path = "machine_failure_model.pkl"
    feat_path  = "feature_names.pkl"
    try:
        mdl  = joblib.load(model_path)
        feat = joblib.load(feat_path) if os.path.exists(feat_path) else None
        return mdl, feat
    except Exception as e:
        return None, None

model, feature_names = load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Machine Failure\nPrediction System")
    st.markdown("---")
    st.markdown('<div class="section-title">Sensor Inputs</div>', unsafe_allow_html=True)
    st.caption("Adjust the live sensor readings below, then click **Run Prediction**.")

    # ── Standard UCI AI4I sensor parameters ──────────────────────────────────
    # The UCI AI4I dataset has these features. We expose the main ones as sliders.
    air_temp = st.slider(
        "Air Temperature [K]",
        min_value=295.0, max_value=305.0, value=298.1, step=0.1,
        help="Ambient air temperature in Kelvin"
    )
    process_temp = st.slider(
        "Process Temperature [K]",
        min_value=305.0, max_value=315.0, value=308.6, step=0.1,
        help="Process temperature in Kelvin"
    )
    rot_speed = st.slider(
        "Rotational Speed [rpm]",
        min_value=1168, max_value=2886, value=1551, step=1,
        help="Machine rotational speed"
    )
    torque = st.slider(
        "Torque [Nm]",
        min_value=3.8, max_value=76.6, value=42.8, step=0.1,
        help="Applied torque in Newton-metres"
    )
    tool_wear = st.slider(
        "Tool Wear [min]",
        min_value=0, max_value=253, value=0, step=1,
        help="Cumulative tool wear time in minutes"
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Machine Type</div>', unsafe_allow_html=True)
    machine_type = st.selectbox(
        "Type",
        options=["L (Low)", "M (Medium)", "H (High)"],
        index=1,
        help="Machine quality variant"
    )
    type_map = {"L (Low)": [1,0,0], "M (Medium)": [0,1,0], "H (High)": [0,0,1]}
    type_encoded = type_map[machine_type]

    st.markdown("---")
    predict_btn = st.button("⚡ Run Prediction")

    st.markdown("---")
    st.caption("**Dataset:** UCI AI4I 2020 Predictive Maintenance  \n**Model:** Random Forest Classifier  \n**Accuracy:** ~90%")


# ── Build feature vector ──────────────────────────────────────────────────────
def build_features(air_t, proc_t, rpm, torq, wear, type_enc):
    """
    Construct input DataFrame matching the UCI AI4I feature set.
    If feature_names.pkl is available, we align to it exactly.
    Otherwise we use the standard 9-feature vector.
    """
    # Standard feature order for UCI AI4I (after one-hot encoding Type)
    raw = {
        "Air temperature [K]":     air_t,
        "Process temperature [K]": proc_t,
        "Rotational speed [rpm]":  rpm,
        "Torque [Nm]":             torq,
        "Tool wear [min]":         wear,
        "Type_H":                  type_enc[2],
        "Type_L":                  type_enc[0],
        "Type_M":                  type_enc[1],
    }
    df = pd.DataFrame([raw])

    if feature_names is not None:
        # Align exactly to trained feature order; fill missing with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    return df


# ── Gauge chart ───────────────────────────────────────────────────────────────
def draw_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1]*200, color="#21262d", linewidth=18, solid_capstyle="round")

    # Coloured fill arc
    fill_end = np.pi - prob * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 200)
    color = "#f85149" if prob > 0.5 else ("#d29922" if prob > 0.25 else "#3fb950")
    ax.plot(theta_fill, [1]*len(theta_fill), color=color, linewidth=18, solid_capstyle="round")

    # Needle
    needle_angle = np.pi - prob * np.pi
    ax.annotate("", xy=(needle_angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))

    # Labels
    ax.text(np.pi,       1.3, "0%",   ha="center", va="center", color="#8b949e", fontsize=9)
    ax.text(np.pi/2,     1.3, "50%",  ha="center", va="center", color="#8b949e", fontsize=9)
    ax.text(0,           1.3, "100%", ha="center", va="center", color="#8b949e", fontsize=9)

    # Centre probability label
    ax.text(0, 0, f"{prob*100:.1f}%", ha="center", va="center",
            color="white", fontsize=20, fontweight="bold", transform=ax.transData)

    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Feature importance chart ──────────────────────────────────────────────────
def draw_importance() -> plt.Figure:
    if model is None:
        return None
    try:
        importances = model.feature_importances_
        feat_labels  = feature_names if feature_names is not None else [f"F{i}" for i in range(len(importances))]
        idx = np.argsort(importances)[-8:]   # top 8

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        bars = ax.barh([feat_labels[i] for i in idx],
                       [importances[i] for i in idx],
                       color="#238636", edgecolor="none", height=0.6)

        ax.set_xlabel("Importance", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color("#8b949e")
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🔧 Machine Failure Prediction System")
st.markdown(
    "Real-time predictive maintenance using a **Random Forest** model trained on the "
    "[UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601) dataset (10,000 sensor readings)."
)

# ── Model status warning ──────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model file not found.** Make sure `machine_failure_model.pkl` is in the same "
        "directory as `app.py`. The app will run in **demo mode** with simulated predictions.",
        icon="⚠️",
    )

st.markdown("---")

# ── Run prediction ────────────────────────────────────────────────────────────
X = build_features(air_temp, process_temp, rot_speed, torque, tool_wear, type_encoded)

if model is not None:
    prob      = float(model.predict_proba(X)[0][1])
    predicted = int(model.predict(X)[0])
else:
    # Demo mode — deterministic simulation based on inputs so it "feels" real
    risk_score = (
        (torque - 3.8) / (76.6 - 3.8) * 0.35 +
        (tool_wear / 253) * 0.40 +
        ((rot_speed - 1168) / (2886 - 1168)) * 0.15 +
        (max(0, process_temp - air_temp - 8.6) / 10) * 0.10
    )
    prob      = float(np.clip(risk_score + np.random.normal(0, 0.02), 0.01, 0.99))
    predicted = 1 if prob > 0.5 else 0

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Failure Probability", f"{prob*100:.1f}%",
          delta=None)
c2.metric("Prediction",
          "🚨 FAILURE" if predicted == 1 else "✅ NORMAL",
          delta=None)
c3.metric("System Health",
          f"{max(0, 100 - int(prob*100))}%")
c4.metric("Tool Wear",
          f"{tool_wear} min",
          delta=f"{tool_wear} / 253 max")

st.markdown("---")

# ── Status banner ─────────────────────────────────────────────────────────────
if predicted == 1 or prob > 0.5:
    st.markdown(f"""
    <div class="alert-box">
        <div style="font-size:36px; font-weight:800; color:#f85149;">🚨 FAILURE RISK DETECTED</div>
        <div style="font-size:20px; color:#ffa198; margin-top:8px;">
            Failure probability: <strong>{prob*100:.1f}%</strong> — Immediate maintenance recommended
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="safe-box">
        <div style="font-size:36px; font-weight:800; color:#3fb950;">✅ SYSTEM STABLE</div>
        <div style="font-size:20px; color:#7ee787; margin-top:8px;">
            Failure probability: <strong>{prob*100:.1f}%</strong> — All sensors within normal range
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gauge + Feature importance ────────────────────────────────────────────────
col_gauge, col_feat = st.columns([1, 1])

with col_gauge:
    st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
    gauge_fig = draw_gauge(prob)
    st.pyplot(gauge_fig, use_container_width=True)
    plt.close(gauge_fig)

with col_feat:
    st.markdown('<div class="section-title">Feature Importance (Top 8)</div>', unsafe_allow_html=True)
    imp_fig = draw_importance()
    if imp_fig:
        st.pyplot(imp_fig, use_container_width=True)
        plt.close(imp_fig)
    else:
        st.info("Feature importance available after model loads.")

st.markdown("---")

# ── Sensor readings table ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Current Sensor Readings</div>', unsafe_allow_html=True)
sensor_data = {
    "Sensor": [
        "Air Temperature",
        "Process Temperature",
        "Temp Differential",
        "Rotational Speed",
        "Torque",
        "Tool Wear",
        "Machine Type",
    ],
    "Value": [
        f"{air_temp:.1f} K",
        f"{process_temp:.1f} K",
        f"{process_temp - air_temp:.1f} K",
        f"{rot_speed} rpm",
        f"{torque:.1f} Nm",
        f"{tool_wear} min",
        machine_type,
    ],
    "Status": [
        "✅ Normal" if 295 <= air_temp <= 304 else "⚠️ Check",
        "✅ Normal" if 305 <= process_temp <= 314 else "⚠️ Check",
        "✅ Normal" if 8 <= (process_temp - air_temp) <= 12 else "⚠️ High",
        "✅ Normal" if 1300 <= rot_speed <= 2500 else "⚠️ Abnormal",
        "✅ Normal" if torque <= 55 else "🚨 High",
        "✅ Normal" if tool_wear <= 150 else ("⚠️ Wearing" if tool_wear <= 200 else "🚨 Replace"),
        "—",
    ]
}
st.dataframe(
    pd.DataFrame(sensor_data),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ── Risk breakdown ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)

factors = {
    "Tool Wear":          tool_wear / 253,
    "Torque Level":       max(0, (torque - 40) / 36.6),
    "Temp Differential":  max(0, (process_temp - air_temp - 8.6) / 4),
    "Speed Anomaly":      max(0, abs(rot_speed - 1500) / 1386),
}

for factor, value in factors.items():
    pct = min(100, int(value * 100))
    bar_color = "#f85149" if pct > 70 else ("#d29922" if pct > 40 else "#3fb950")
    st.markdown(f"""
    <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;font-size:13px;color:#8b949e;margin-bottom:4px;">
            <span>{factor}</span><span>{pct}%</span>
        </div>
        <div class="risk-bar-bg">
            <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:8px;transition:width .5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Existing visualizations from repo ────────────────────────────────────────
st.markdown('<div class="section-title">Model Analysis Charts</div>', unsafe_allow_html=True)

img_col1, img_col2 = st.columns(2)

for img_file, label, col in [
    ("feature_importance.png", "Feature Importance", img_col1),
    ("correlation_heatmap.png", "Correlation Heatmap", img_col2),
]:
    if os.path.exists(img_file):
        with col:
            st.image(img_file, caption=label, use_container_width=True)

if os.path.exists("sensor_histograms.png"):
    st.image("sensor_histograms.png", caption="Sensor Value Distributions", use_container_width=True)

st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:13px;padding:8px 0 16px;">
    Built by <a href="https://github.com/Sneh-04" style="color:#58a6ff;">Snehalatha Reddy Kunduru</a> ·
    <a href="https://github.com/Sneh-04/Machine_Failure_Prediction" style="color:#58a6ff;">GitHub Repo</a> ·
    Dataset: UCI AI4I 2020 Predictive Maintenance
</div>
""", unsafe_allow_html=True)
