import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

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

.app-title{
    font-family: 'Orbitron', sans-serif;
    font-size: 48px;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.glass{
    background: var(--glass-bg);
    border-radius: 16px;
    padding: 18px;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.03);
}
.status-card{border-radius: 18px; padding: 28px; text-align: center;}
.status-ok{box-shadow: 0 8px 30px rgba(0,255,242,0.06); border: 1px solid rgba(0,255,242,0.06);}
.status-alert{
    box-shadow: 0 8px 40px rgba(255,20,85,0.08);
    border: 1px solid rgba(255,20,85,0.12);
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0%  {transform: translateY(0)}
    50% {transform: translateY(-2px)}
    100%{transform: translateY(0)}
}
.big-risk{font-size: 54px; font-weight: 800;}
.small-sub{font-size: 20px; opacity: 0.8;}
.kpi{text-align: center;}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model and feature names if available."""
    model = None
    feature_names = None
    if os.path.exists("machine_failure_model.pkl"):
        try:
            model = joblib.load("machine_failure_model.pkl")
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    if os.path.exists("feature_names.pkl"):
        try:
            feature_names = joblib.load("feature_names.pkl")
        except Exception as e:
            st.warning(f"Could not load feature names: {e}")
    return model, feature_names

model, feature_names = load_model()

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">AI Predictive Maintenance System</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Query-param forced alert (for screenshots / testing) ──────────────────────
# FIX: st.experimental_get_query_params() removed in Streamlit ≥1.34 → use st.query_params
try:
    qparams = st.query_params
    if qparams.get("force_alert") in ("1", "true", "True"):
        st.session_state.last_risk = 0.92
        st.session_state.last_status = "CRITICAL FAILURE"
except Exception:
    pass

# ── Status Card ────────────────────────────────────────────────────────────────
status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
with status_col2:
    risk_pct   = st.session_state.get("last_risk", 0.12)
    status_text = st.session_state.get("last_status", "SYSTEM STABLE")
    status_class = "status-alert" if isinstance(risk_pct, float) and risk_pct > 0.5 else "status-ok"

    st.markdown(f'<div class="glass status-card {status_class}">', unsafe_allow_html=True)
    st.markdown(f"<div class='big-risk'>{status_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-sub'>Failure Risk: <strong>{int(risk_pct * 100)}%</strong></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
last_prob     = st.session_state.get("last_risk", 0.12)
ai_conf       = int((1 - abs(0.5 - last_prob)) * 100)
system_health = max(0, 100 - int(last_prob * 100))
sensor_load   = st.session_state.get("sensor_load", 27)
risk_level    = "Low" if last_prob < 0.25 else ("Medium" if last_prob < 0.5 else "High")

k1.metric("System Health", f"{system_health}%")
k2.metric("AI Confidence", f"{ai_conf}%")
k3.metric("Sensor Load",   f"{sensor_load}%")
k4.metric("Risk Level",    risk_level)

st.markdown("---")

# ── Main Area ──────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Live Sensor Graph")
    chart_placeholder = st.empty()

    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame({"value": np.zeros(40)})

    chart_placeholder.line_chart(st.session_state.chart_data)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Input Controls")

    # Force-alert toggle for previewing alert state
    force_alert = st.checkbox("Force Alert (preview)", value=False)
    if force_alert:
        st.session_state.last_risk   = 0.92
        st.session_state.last_status = "CRITICAL FAILURE"

    with st.expander("Sensor Inputs (compact)"):
        cols = st.columns(2)
        footfall    = cols[0].slider("Footfall",    0, 100, int(st.session_state.get("footfall",    30)))
        temperature = cols[1].slider("Temperature", 0, 150, int(st.session_state.get("temperature", 60)))
        vibration   = cols[0].slider("Vibration",   0, 100, int(st.session_state.get("vibration",   20)))
        load        = cols[1].slider("Load",        0, 100, int(st.session_state.get("load",        25)))

        # FIX: expose previously-hidden features so the model gets real inputs.
        # These replace the four hardcoded zeros in the original code.
        AQ          = st.number_input("Air Quality (AQ)",       min_value=0,   max_value=500, value=int(st.session_state.get("AQ",         100)))
        rpm         = st.number_input("RPM",                    min_value=0,   max_value=5000, value=int(st.session_state.get("rpm",       1500)))
        pressure    = st.number_input("Pressure (bar)",         min_value=0,   max_value=200, value=int(st.session_state.get("pressure",    50)))
        humidity    = st.number_input("Humidity (%)",           min_value=0,   max_value=100, value=int(st.session_state.get("humidity",    45)))
        power       = st.number_input("Power draw (kW)",        min_value=0.0, max_value=500.0, value=float(st.session_state.get("power",  10.0)))

    # Persist to session state
    for k, v in dict(footfall=footfall, temperature=temperature, vibration=vibration,
                     load=load, AQ=AQ, rpm=rpm, pressure=pressure,
                     humidity=humidity, power=power).items():
        st.session_state[k] = v

    # FIX: live-update is done one step at a time via rerun, not a blocking sleep loop.
    # Buttons toggle a flag; a single step happens each rerun while active.
    col_start, col_stop = st.columns(2)
    start_live = col_start.button("▶ Start Live")
    stop_live  = col_stop.button("■ Stop")
    run_scan   = st.button("⚡ INITIATE AI SCAN", use_container_width=True)

    if start_live:
        st.session_state.updating = True
    if stop_live:
        st.session_state.updating = False

    st.markdown('</div>', unsafe_allow_html=True)

# ── Non-blocking live update (one step per rerun) ─────────────────────────────
if st.session_state.get("updating", False):
    current = st.session_state.chart_data["value"].iloc[-1]
    new_val = float(np.clip(
        np.random.normal(loc=0.2 * (st.session_state.get("load", 25) / 25), scale=0.5),
        -3, 3
    )) + current

    st.session_state.chart_data = pd.concat(
        [st.session_state.chart_data.iloc[1:], pd.DataFrame({"value": [new_val]})],
        ignore_index=True
    )
    chart_placeholder.line_chart(st.session_state.chart_data)
    st.session_state.sensor_load = int(np.clip(abs(new_val) * 10, 5, 95))

    # FIX: use st.rerun() — st.experimental_rerun() removed in Streamlit ≥1.34
    st.rerun()

# ── AI Scan ────────────────────────────────────────────────────────────────────
if run_scan:
    # FIX: all 9 features now populated from real user inputs (no more hardcoded zeros).
    # Order: footfall, vibration, AQ, load, rpm, pressure, humidity, power, temperature
    # If feature_names.pkl is available, we validate the expected count.
    feature_values = [
        st.session_state.get("footfall",    30),
        st.session_state.get("vibration",   20),
        st.session_state.get("AQ",         100),
        st.session_state.get("load",        25),
        st.session_state.get("rpm",       1500),
        st.session_state.get("pressure",    50),
        st.session_state.get("humidity",    45),
        st.session_state.get("power",     10.0),
        st.session_state.get("temperature", 60),
    ]

    if feature_names is not None and len(feature_names) != len(feature_values):
        st.error(
            f"Feature mismatch: model expects {len(feature_names)} features "
            f"but {len(feature_values)} were provided. "
            f"Expected: {list(feature_names)}"
        )
    else:
        features = np.array([feature_values])

        if model is not None:
            pred = model.predict(features)[0]
            prob = float(model.predict_proba(features)[0][1])
        else:
            pred = 0
            prob = float(np.clip(np.random.beta(2, 18), 0.01, 0.95))

        st.session_state.last_risk   = prob
        st.session_state.last_status = "CRITICAL FAILURE" if prob > 0.5 else "SYSTEM STABLE"

        if pred == 1 or prob > 0.5:
            st.markdown('<div class="glass status-card status-alert">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;color:#ff6aa6'>🚨 CRITICAL FAILURE IMMINENT</h2>",
                        unsafe_allow_html=True)
            # FIX: missing space between <p and style= fixed
            st.markdown(f"<p style='text-align:center;'>Failure Probability: <strong>{prob*100:.2f}%</strong></p>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass status-card status-ok">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;color:#7fffd4'>✅ SYSTEM STABLE</h2>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Failure Probability: <strong>{prob*100:.2f}%</strong></p>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # FIX: use st.rerun() — st.experimental_rerun() removed in Streamlit ≥1.34
        st.rerun()
