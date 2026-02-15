import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd

st.set_page_config(layout="wide", page_title="AI Predictive Maintenance")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
:root{--glass-bg: rgba(255,255,255,0.04); --accent: #00f7ff; --accent-2: #ff00d0;}
html, body {
  background: #071025;
  color: #e6f7ff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.app-title{font-family: 'Orbitron', sans-serif; font-size:48px; text-align:center; font-weight:700; background:linear-gradient(90deg,var(--accent-2),var(--accent)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:6px}
.glass{background: var(--glass-bg); border-radius:16px; padding:18px; backdrop-filter: blur(6px); border:1px solid rgba(255,255,255,0.03)}
.status-card{border-radius:18px; padding:28px; text-align:center;}
.status-ok{box-shadow:0 8px 30px rgba(0,255,242,0.06); border:1px solid rgba(0,255,242,0.06)}
.status-alert{box-shadow:0 8px 40px rgba(255,20,85,0.08); border:1px solid rgba(255,20,85,0.12); animation: pulse 1.6s infinite;}
@keyframes pulse {0%{transform:translateY(0)}50%{transform:translateY(-2px)}100%{transform:translateY(0)}}
.big-risk{font-size:54px; font-weight:800}
.small-sub{font-size:20px; opacity:0.8}
.kpi {text-align:center}
</style>
""", unsafe_allow_html=True)

try:
    model = joblib.load("machine_failure_model.pkl")
except Exception:
    model = None

st.markdown('<div class="app-title">AI Predictive Maintenance System</div>', unsafe_allow_html=True)
st.markdown("---")

# Read URL query param to force alert for screenshots or testing (guard for Streamlit versions)
try:
    qparams = st.experimental_get_query_params()
    if qparams.get("force_alert", [None])[0] in ("1", "true", "True"):
        st.session_state.last_risk = 0.92
        st.session_state.last_status = "CRITICAL FAILURE"
except AttributeError:
    # older Streamlit may not expose experimental_get_query_params; ignore gracefully
    qparams = {}

# Top: Centered Status Card
status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
with status_col2:
    st.markdown('<div class="glass status-card status-ok" id="status-card">', unsafe_allow_html=True)
    risk_pct = st.session_state.get("last_risk", 0.12)
    status_text = st.session_state.get("last_status", "SYSTEM STABLE")
    status_class = "status-ok"
    if isinstance(risk_pct, float) and risk_pct > 0.5:
        status_text = "CRITICAL FAILURE"
        status_class = "status-alert"
    st.markdown(f'<div class="{status_class}">', unsafe_allow_html=True)
    st.markdown(f"<div class='big-risk'>{status_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-sub'>Failure Risk: <strong>{int(risk_pct*100)}%</strong></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# KPI Cards
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    # derive some placeholder KPI values
    last_prob = st.session_state.get("last_risk", 0.12)
    ai_conf = int((1 - abs(0.5 - last_prob)) * 100)
    system_health = max(0, 100 - int(last_prob * 100))
    sensor_load = st.session_state.get("sensor_load", 27)
    risk_level = "Low" if last_prob < 0.25 else ("Medium" if last_prob < 0.5 else "High")
    k1.metric("System Health", f"{system_health}%")
    k2.metric("AI Confidence", f"{ai_conf}%")
    k3.metric("Sensor Load", f"{sensor_load}%")
    k4.metric("Risk Level", risk_level)

st.markdown("---")

# Main area: Live Chart + Controls
left, right = st.columns([3, 1])

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Live Sensor Graph")
    chart_placeholder = st.empty()
    # prepare initial data
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame({"value": np.zeros(40)})

    chart = chart_placeholder.line_chart(st.session_state.chart_data)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Input Controls")
    # Provide a Force Alert toggle for previewing alert state (useful for screenshots)
    force_alert = st.checkbox("Force Alert (preview)", value=False)
    if force_alert:
        st.session_state.last_risk = 0.92
        st.session_state.last_status = "CRITICAL FAILURE"
    elif "last_risk" in st.session_state and st.session_state.get("last_risk") is not None and st.session_state.get("last_risk") < 0.9:
        # do nothing, keep last known
        pass
    with st.expander("Sensor Inputs (compact)"):
        cols = st.columns(2)
        footfall = cols[0].slider("Footfall", 0, 100, int(st.session_state.get("footfall", 30)))
        temperature = cols[1].slider("Temperature", 0, 150, int(st.session_state.get("temperature", 60)))
        vibration = cols[0].slider("Vibration", 0, 100, int(st.session_state.get("vibration", 20)))
        load = cols[1].slider("Load", 0, 100, int(st.session_state.get("load", 25)))
        # keep a compact set but preserve original full feature vector inputs hidden
        AQ = st.number_input("Air Quality (AQ)", min_value=0, max_value=500, value=int(st.session_state.get("AQ", 100)))
        # Save into session for small demo continuity
        st.session_state.footfall = footfall
        st.session_state.temperature = temperature
        st.session_state.vibration = vibration
        st.session_state.load = load
        st.session_state.AQ = AQ

    start_live = st.button("▶ Start Live")
    stop_live = st.button("■ Stop")
    run_scan = st.button("⚡ INITIATE AI SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

# Live update loop (runs for a short demo when Start Live pressed)
if start_live:
    st.session_state.updating = True

if stop_live:
    st.session_state.updating = False

if st.session_state.get("updating", False):
    for _ in range(60):
        new = float(np.clip(np.random.normal(loc=0.2 * (st.session_state.get('load', 25)/25), scale=0.5), -3, 3)) + st.session_state.chart_data['value'].iloc[-1]
        st.session_state.chart_data = pd.concat([st.session_state.chart_data.iloc[1:], pd.DataFrame({"value": [new]})], ignore_index=True)
        chart_placeholder.line_chart(st.session_state.chart_data)
        st.session_state.sensor_load = int(np.clip(abs(new) * 10, 5, 95))
        time.sleep(0.08)

# Run AI scan and update status
if run_scan:
    # Build a features vector compatible with original model shape (9 features)
    features = np.array([[st.session_state.get('footfall',30),
                          st.session_state.get('vibration',20),
                          st.session_state.get('AQ',100),
                          st.session_state.get('load',25),
                          0, 0, 0, 0, st.session_state.get('temperature',60)]])
    if model is not None:
        pred = model.predict(features)[0]
        prob = float(model.predict_proba(features)[0][1])
    else:
        pred = 0
        prob = float(np.clip(np.random.beta(2,18), 0.01, 0.95))

    st.session_state.last_risk = prob
    st.session_state.last_status = ("CRITICAL FAILURE" if prob > 0.5 else "SYSTEM STABLE")
    st.session_state.sensor_load = st.session_state.get('sensor_load', 25)

    # show result area
    if pred == 1 or prob > 0.5:
        st.markdown('<div class="glass status-card status-alert">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;color:#ff6aa6'>🚨 CRITICAL FAILURE IMMINENT</h2>", unsafe_allow_html=True)
        st.markdown(f"<pstyle='text-align:center;'>Failure Probability: <strong>{prob*100:.2f}%</strong></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass status-card status-ok">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;color:#7fffd4'>✅ SYSTEM STABLE</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Failure Probability: <strong>{prob*100:.2f}%</strong></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.experimental_rerun()
