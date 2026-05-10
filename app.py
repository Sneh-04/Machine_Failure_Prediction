# ================================
# MACHINEGUARD AI — FINAL CLEAN VERSION
# ================================
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import json
 
# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide"
)
 
# ─────────────────────────────────────
# COLORS
# ─────────────────────────────────────
PRIMARY = "#00E0FF"
WARNING = "#FF8C42"
DANGER  = "#FF4D4D"
SUCCESS = "#00D26A"
 
BG      = "#0B0F14"
CARD    = "#121821"
TEXT    = "#D6E2F0"
 
# ─────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────
st.markdown(f"""
<style>
 
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
 
html, body, .stApp {{
    background:
        radial-gradient(circle at top right,
        rgba(0,224,255,0.06),
        transparent 30%),
        {BG};
 
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}
 
.block-container {{
    max-width: 1450px;
    padding-top: 1.8rem;
}}
 
section[data-testid="stSidebar"] {{
    background: #10161F;
    border-right: 1px solid rgba(255,255,255,0.05);
}}
 
.metric-card {{
    background: rgba(18,24,33,0.88);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}}
 
.glass {{
    background: rgba(18,24,33,0.88);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(10px);
}}
 
.main-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 42px;
    font-weight: 900;
    color: white;
}}
 
.sub-title {{
    color: #7D8CA3;
    margin-top: -8px;
}}
 
.status-normal {{
    background: rgba(0,210,106,0.12);
    border: 1px solid rgba(0,210,106,0.3);
    border-radius: 18px;
    padding: 20px;
}}
 
.status-danger {{
    background: rgba(255,77,77,0.12);
    border: 1px solid rgba(255,77,77,0.3);
    border-radius: 18px;
    padding: 20px;
    animation: pulse 1.5s infinite;
}}
 
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 rgba(255,77,77,0.2); }}
    50% {{ box-shadow: 0 0 30px rgba(255,77,77,0.35); }}
    100% {{ box-shadow: 0 0 0 rgba(255,77,77,0.2); }}
}}
 
.stButton button {{
    width: 100%;
    height: 48px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, {PRIMARY}, #0099ff);
    color: black;
    font-weight: 700;
}}
 
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────
@st.cache_resource
def load_model():
 
    try:
        model = joblib.load("machine_failure_model.pkl")
        features = joblib.load("feature_names.pkl")
        return model, features
 
    except Exception as e:
 
        st.warning(f"Model not loaded: {e}")
        return None, None
 
 
# ─────────────────────────────────────
# LOAD METRICS
# ─────────────────────────────────────
def load_metrics():
 
    default = {
        "Accuracy": "—",
        "Precision": "—",
        "Recall": "—",
        "F1 Score": "—",
        "ROC-AUC": "—"
    }
 
    if os.path.exists("metrics.json"):
 
        try:
 
            with open("metrics.json") as f:
                data = json.load(f)
 
            return {
                k: f"{v*100:.1f}%"
                for k, v in data.items()
            }
 
        except Exception:
            return default
 
    return default
 
 
model, feature_names = load_model()
perf_metrics = load_metrics()
 
# ─────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────
if "sim_tick" not in st.session_state:
    st.session_state.sim_tick = 0
 
# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
with st.sidebar:
 
    st.markdown("""
    <div style='margin-bottom:20px'>
        <div style='font-size:28px;font-weight:800;color:white'>
            ⚙ MachineGuard
        </div>
 
        <div style='color:#7D8CA3;font-size:13px'>
            Industrial AI Monitoring System
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    live_mode = st.toggle(
        "Enable Live Simulation",
        value=False
    )
 
    st.subheader("Sensor Inputs")
 
    air_temp = st.slider(
        "Air Temperature [K]",
        295.0, 305.0, 298.1
    )
 
    process_temp = st.slider(
        "Process Temperature [K]",
        305.0, 315.0, 308.6
    )
 
    rot_speed = st.slider(
        "Rotational Speed [rpm]",
        1168, 2886, 1551
    )
 
    torque = st.slider(
        "Torque [Nm]",
        3.8, 76.6, 42.8
    )
 
    tool_wear = st.slider(
        "Tool Wear [min]",
        0, 253, 0
    )
 
    machine_type = st.selectbox(
        "Machine Type",
        ["Low", "Medium", "High"]
    )
 
# ─────────────────────────────────────
# LIVE SIMULATION
# ─────────────────────────────────────
rng = np.random.default_rng(
    seed=st.session_state.sim_tick
)
 
if live_mode:
 
    st.session_state.sim_tick += 1
 
    sim_air = air_temp + rng.normal(0, 0.2)
    sim_proc = process_temp + rng.normal(0, 0.3)
    sim_rpm = rot_speed + int(rng.normal(0, 20))
    sim_torque = torque + rng.normal(0, 1.2)
    sim_wear = tool_wear
 
else:
 
    sim_air = air_temp
    sim_proc = process_temp
    sim_rpm = rot_speed
    sim_torque = torque
    sim_wear = tool_wear
 
# ─────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────
type_map = {
    "Low": {"Type_H":0,"Type_L":1,"Type_M":0},
    "Medium": {"Type_H":0,"Type_L":0,"Type_M":1},
    "High": {"Type_H":1,"Type_L":0,"Type_M":0}
}
 
data = pd.DataFrame([{
    "Air temperature [K]": sim_air,
    "Process temperature [K]": sim_proc,
    "Rotational speed [rpm]": sim_rpm,
    "Torque [Nm]": sim_torque,
    "Tool wear [min]": sim_wear,
    **type_map[machine_type]
}])
 
if feature_names is not None:
 
    for col in feature_names:
 
        if col not in data.columns:
            data[col] = 0
 
    data = data[feature_names]
 
# ─────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────
if model is not None:
 
    prob = float(
        model.predict_proba(data)[0][1]
    )
 
    prediction = int(
        model.predict(data)[0]
    )
 
else:
 
    risk = (
        (sim_torque / 76.6) * 0.30 +
        (sim_wear / 253) * 0.40 +
        ((sim_proc - sim_air) / 15) * 0.30
    )
 
    prob = min(max(risk, 0.05), 0.95)
    prediction = 1 if prob > 0.5 else 0
 
# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
left, right = st.columns([4,1])
 
with left:
 
    st.markdown("""
    <div class='main-title'>
        ⚙ MachineGuard AI
    </div>
 
    <div class='sub-title'>
        Real-Time Predictive Maintenance & Industrial Anomaly Detection Platform
    </div>
    """, unsafe_allow_html=True)
 
with right:
 
    status = (
        "Operational"
        if prediction == 0
        else "Failure Risk"
    )
 
    color = (
        SUCCESS
        if prediction == 0
        else DANGER
    )
 
    st.markdown(f"""
    <div class='glass' style='text-align:center'>
        <div style='font-size:12px;color:#7D8CA3'>
            Machine Status
        </div>
 
        <div style='font-size:24px;
                    color:{color};
                    font-weight:700'>
            {status}
        </div>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
 
metrics = [
    ("Failure Risk", f"{prob*100:.1f}%"),
    ("System Health", f"{100-int(prob*100)}%"),
    ("Tool Wear", f"{sim_wear} min"),
    ("Temperature Delta", f"{sim_proc-sim_air:.1f} K"),
    ("Machine Speed", f"{sim_rpm} rpm")
]
 
for col, (title, value) in zip(
    [c1,c2,c3,c4,c5],
    metrics
):
 
    with col:
 
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#7D8CA3;font-size:13px'>
                {title}
            </div>
 
            <div style='font-size:30px;
                        color:white;
                        font-weight:700'>
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# STATUS ALERT
# ─────────────────────────────────────
if prediction == 0:
 
    st.markdown(f"""
    <div class='status-normal'>
        <div style='font-size:24px;
                    color:{SUCCESS};
                    font-weight:700'>
            ✅ Predictive Maintenance Status: Stable
        </div>
 
        <div style='margin-top:8px;color:#9FB2C7'>
            AI monitoring indicates all operational parameters are
            within expected industrial thresholds.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
else:
 
    st.markdown(f"""
    <div class='status-danger'>
        <div style='font-size:24px;
                    color:{DANGER};
                    font-weight:700'>
            🚨 Industrial Anomaly Detected
        </div>
 
        <div style='margin-top:8px;color:#9FB2C7'>
            Sensor telemetry suggests elevated machine failure probability.
            Preventive maintenance intervention recommended immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# MAIN CHARTS
# ─────────────────────────────────────
g1, g2 = st.columns([1,1])
 
with g1:
 
    st.markdown("### Live Failure Risk Gauge")
 
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
 
        number={
            "suffix":"%",
            "font":{
                "size":42,
                "color":"white"
            }
        },
 
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":PRIMARY},
 
            "steps":[
                {
                    "range":[0,35],
                    "color":"rgba(0,210,106,0.35)"
                },
                {
                    "range":[35,70],
                    "color":"rgba(255,140,66,0.35)"
                },
                {
                    "range":[70,100],
                    "color":"rgba(255,77,77,0.35)"
                }
            ],
 
            "bgcolor":CARD
        }
    ))
 
    gauge.update_layout(
        paper_bgcolor=BG,
        font={"color":"white"},
        height=360
    )
 
    st.plotly_chart(
        gauge,
        width="stretch"
    )
 
with g2:
 
    st.markdown("### Model Feature Importance")
 
    if model is not None and hasattr(model, "feature_importances_"):
 
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })
 
        importance = importance.sort_values(
            by="Importance",
            ascending=True
        )
 
        fig = px.bar(
            importance.tail(8),
            x="Importance",
            y="Feature",
            orientation='h',
            color="Importance",
            color_continuous_scale=[
                [0, "#0099ff"],
                [1, PRIMARY]
            ]
        )
 
        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font_color="white",
            height=360,
            coloraxis_showscale=False
        )
 
        st.plotly_chart(
            fig,
            width="stretch"
        )
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# MODEL METRICS
# ─────────────────────────────────────
st.markdown("### Model Performance Metrics")
 
if "—" in perf_metrics.values():
 
    st.caption("""
⚠️ metrics.json not found — run python save_metrics.py
after training to display real computed values here.
""")
 
m1,m2,m3,m4,m5 = st.columns(5)
 
for col, (name, value) in zip(
    [m1,m2,m3,m4,m5],
    perf_metrics.items()
):
 
    with col:
 
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#7D8CA3;font-size:13px'>
                {name}
            </div>
 
            <div style='font-size:28px;
                        color:{PRIMARY};
                        font-weight:700'>
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────
# LIVE TELEMETRY
# ─────────────────────────────────────
st.markdown("### Live Industrial Telemetry")
 
telemetry = pd.DataFrame({
 
    "Sensor":[
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
 
    "Current Reading":[
        f"{sim_air:.1f} K",
        f"{sim_proc:.1f} K",
        f"{sim_rpm} rpm",
        f"{sim_torque:.1f} Nm",
        f"{sim_wear} min"
    ],
 
    "Status":[
        "Stable",
        "Monitoring",
        "Operational",
        "Analyzing",
        "Healthy"
    ]
})
 
st.dataframe(
    telemetry,
    width="stretch",
    hide_index=True
)
 
# ─────────────────────────────────────
# AI INSIGHTS
# ─────────────────────────────────────
st.markdown("### 🤖 AI Failure Explanation")
 
explanations = []
 
if sim_torque > 55:
 
    explanations.append(
        "Elevated torque load indicates abnormal mechanical stress."
    )
 
if sim_wear > 180:
 
    explanations.append(
        "Tool wear approaching maintenance threshold."
    )
 
if sim_proc - sim_air > 10:
 
    explanations.append(
        "Thermal differential anomaly detected between process and air temperatures."
    )
 
if sim_rpm < 1300:
 
    explanations.append(
        "Rotational instability may affect operational efficiency."
    )
 
if prediction == 1:
 
    explanations.append(
        "Machine learning model predicts increasing probability of component degradation."
    )
 
if not explanations:
 
    explanations.append(
        "AI diagnostics indicate stable operational performance across all monitored parameters."
    )
 
for item in explanations:
 
    st.markdown(f"""
    <div class='glass'
         style='margin-bottom:12px'>
        {item}
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────
# SYSTEM ARCHITECTURE
# ─────────────────────────────────────
st.markdown("### System Architecture")
 
nodes = [
    "IoT Sensors",
    "Preprocessing",
    "Feature Engineering",
    "Random Forest",
    "Prediction Engine",
    "Dashboard"
]
 
colors = [
    "#00E0FF",
    "#0099ff",
    "#4477ff",
    "#7744ff",
    "#ff4477",
    "#00D26A"
]
 
sankey = go.Figure(go.Sankey(
 
    arrangement="snap",
 
    node=dict(
        pad=20,
        thickness=24,
 
        line=dict(
            color="rgba(255,255,255,0.1)",
            width=0.5
        ),
 
        label=nodes,
        color=colors
    ),
 
    link=dict(
        source=[0,1,2,3,4],
        target=[1,2,3,4,5],
        value=[8,8,8,8,8]
    )
))
 
sankey.update_layout(
    paper_bgcolor=BG,
    font=dict(
        color="white",
        size=14
    ),
    height=320
)
 
st.plotly_chart(
    sankey,
    width="stretch"
)
 
# ─────────────────────────────────────
# ADVANCED ANALYTICS
# ─────────────────────────────────────
with st.expander("Advanced Analytics"):
 
    for img in [
        "feature_importance.png",
        "correlation_heatmap.png",
        "sensor_histograms.png"
    ]:
 
        if os.path.exists(img):
 
            st.image(
                img,
                width="stretch"
            )
 
        else:
 
            st.caption(
                f"{img} not found — generate from training notebook."
            )
 
# ─────────────────────────────────────
# FOOTER
# ─────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
 
st.markdown(f"""
<hr style='border:1px solid rgba(255,255,255,0.06)'>
 
<div style='display:flex;
            justify-content:space-between;
            color:#7D8CA3;
            font-size:13px'>
 
<div>
MachineGuard AI • Industrial Predictive Maintenance • Random Forest Classifier
</div>
 
<div>
<a href='https://github.com/Sneh-04/Machine_Failure_Prediction'
style='color:{PRIMARY};text-decoration:none'>
GitHub
</a>
</div>
 
</div>
""", unsafe_allow_html=True)
 
