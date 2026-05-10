import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MASTER CSS — Industrial Cyberpunk HUD ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #0a0c0f !important;
    color: #c8d6df !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Hex-grid texture background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(255,140,0,0.07) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 66L0 50V16L28 0l28 16v34zm0 0l28 16v18L28 116 0 100V82z' fill='none' stroke='%23131c28' stroke-width='0.6'/%3E%3C/svg%3E");
    background-size: auto, 56px 100px;
    pointer-events: none;
    z-index: 0;
}

/* Scanlines overlay */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px
    );
    pointer-events: none;
    z-index: 1;
}

/* Main content layer */
.main .block-container {
    position: relative;
    z-index: 2;
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #080c12 100%) !important;
    border-right: 1px solid #1a2535 !important;
    position: relative;
    z-index: 2;
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff8c00, #ffb347, transparent);
}

/* ── Sliders ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: #ff8c00 !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: #1a2535 !important;
}
.stSidebar label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    color: #4a6a85 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 14px 20px !important;
    background: transparent !important;
    color: #ff8c00 !important;
    border: 1px solid #ff8c00 !important;
    border-radius: 3px !important;
    clip-path: polygon(10px 0%, 100% 0%, calc(100% - 10px) 100%, 0% 100%);
    transition: all 0.25s ease !important;
    position: relative !important;
}
.stButton > button:hover {
    background: rgba(255,140,0,0.12) !important;
    box-shadow: 0 0 24px rgba(255,140,0,0.35), inset 0 0 16px rgba(255,140,0,0.06) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0d1520 !important;
    border: 1px solid #1e2d40 !important;
    color: #c8d6df !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1520 0%, #080d14 100%) !important;
    border: 1px solid #1a2535 !important;
    border-top: 2px solid #ff8c00 !important;
    border-radius: 5px !important;
    padding: 16px 18px !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 50px; height: 50px;
    background: linear-gradient(135deg, transparent 50%, rgba(255,140,0,0.05) 50%);
}
div[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    color: #3a5570 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #ff8c00 !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1a2535 !important; border-radius: 6px !important; }
.stDataFrame th {
    background: #0c1520 !important;
    color: #ff8c00 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid #1a2535 !important;
}
.stDataFrame td {
    color: #8aacbf !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    border-bottom: 1px solid #0f1822 !important;
    background: #080c12 !important;
}

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
    background: #0d1520 !important;
    border: 1px solid #ff8c00 !important;
    border-left: 3px solid #ff8c00 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #08090c; }
::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #ff8c00; }

/* ── Custom components ── */
.hud-title {
    font-family: 'Orbitron', monospace;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ff8c00;
    text-shadow: 0 0 30px rgba(255,140,0,0.5), 0 0 60px rgba(255,140,0,0.2);
    line-height: 1;
}
.hud-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #2a4060;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 5px;
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #00cc44;
    border-radius: 50%;
    box-shadow: 0 0 8px #00cc44, 0 0 16px rgba(0,204,68,0.4);
    animation: pulse-dot 1.5s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse-dot {
    0%,100%{opacity:1;box-shadow:0 0 8px #00cc44,0 0 16px rgba(0,204,68,.4)}
    50%{opacity:0.5;box-shadow:0 0 4px #00cc44}
}
.hud-divider {
    height: 1px;
    background: linear-gradient(90deg, #ff8c0060, #1e2d40 70%, transparent);
    margin: 14px 0;
}
.sec-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2a4060;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1a2535, transparent);
}
.banner-alert {
    background: linear-gradient(135deg, #1c0600 0%, #0f0300 100%);
    border: 1px solid #ff4500;
    border-left: 4px solid #ff4500;
    border-radius: 5px;
    padding: 18px 24px;
    position: relative;
    overflow: hidden;
    clip-path: polygon(14px 0%,100% 0%,calc(100% - 14px) 100%,0% 100%);
}
.banner-alert::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        45deg,transparent,transparent 5px,rgba(255,69,0,.025) 5px,rgba(255,69,0,.025) 10px
    );
}
.banner-alert-title {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 900;
    color: #ff4500;
    letter-spacing: 0.1em;
    text-shadow: 0 0 20px rgba(255,69,0,.7);
    animation: flicker 3s infinite;
    position: relative;
}
@keyframes flicker {
    0%,100%{opacity:1}91%{opacity:1}92%{opacity:.6}93%{opacity:1}96%{opacity:.8}97%{opacity:1}
}
.banner-alert-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #ff8060;
    margin-top: 5px;
    position: relative;
}
.banner-safe {
    background: linear-gradient(135deg, #001a08 0%, #000d04 100%);
    border: 1px solid #00cc44;
    border-left: 4px solid #00cc44;
    border-radius: 5px;
    padding: 18px 24px;
    clip-path: polygon(14px 0%,100% 0%,calc(100% - 14px) 100%,0% 100%);
}
.banner-safe-title {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 900;
    color: #00cc44;
    letter-spacing: 0.1em;
    text-shadow: 0 0 20px rgba(0,204,68,.5);
}
.banner-safe-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #60cc80;
    margin-top: 5px;
}
.riskbar-wrap { margin-bottom: 13px; }
.riskbar-head {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #3a5570;
    margin-bottom: 4px;
}
.riskbar-track {
    height: 7px;
    background: #0c1520;
    border-radius: 2px;
    border: 1px solid #162030;
    overflow: hidden;
}
.riskbar-fill {
    height: 100%;
    border-radius: 2px;
    position: relative;
}
.riskbar-fill::after {
    content:'';
    position:absolute;
    top:0;right:0;
    width:6px;height:100%;
    background:rgba(255,255,255,.35);
    filter:blur(3px);
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        mdl  = joblib.load("machine_failure_model.pkl")
        feat = joblib.load("feature_names.pkl") if os.path.exists("feature_names.pkl") else None
        return mdl, feat
    except Exception:
        return None, None

model, feature_names = load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:6px 0 18px;">
        <div style="font-family:'Orbitron',monospace;font-size:15px;font-weight:900;
                    color:#ff8c00;letter-spacing:.18em;
                    text-shadow:0 0 15px rgba(255,140,0,.45);">
            ⚙ MACHINEGUARD
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:#2a4060;letter-spacing:.18em;margin-top:3px;">
            PREDICTIVE MAINTENANCE // v2.0
        </div>
        <div style="height:1px;background:linear-gradient(90deg,#ff8c00,transparent);
                    margin:12px 0 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#2a4060;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;">SENSOR INPUTS</div>', unsafe_allow_html=True)

    air_temp     = st.slider("AIR TEMPERATURE [K]",      295.0, 305.0, 298.1, 0.1)
    process_temp = st.slider("PROCESS TEMPERATURE [K]",  305.0, 315.0, 308.6, 0.1)
    rot_speed    = st.slider("ROTATIONAL SPEED [rpm]",   1168,  2886,  1551,  1)
    torque       = st.slider("TORQUE [Nm]",               3.8,   76.6,  42.8,  0.1)
    tool_wear    = st.slider("TOOL WEAR [min]",           0,     253,   0,     1)

    st.markdown('<br><div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#2a4060;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;">MACHINE TYPE</div>', unsafe_allow_html=True)
    machine_type = st.selectbox("", ["L — Low Tolerance", "M — Medium Tolerance", "H — High Precision"], index=1)

    type_map = {
        "L — Low Tolerance":    [1,0,0],
        "M — Medium Tolerance": [0,1,0],
        "H — High Precision":   [0,0,1],
    }
    type_encoded = type_map[machine_type]

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⚡  RUN DIAGNOSTIC SCAN")

    st.markdown("""
    <div style="margin-top:20px;padding-top:14px;border-top:1px solid #1a2535;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:#1e3050;line-height:2.1;letter-spacing:.06em;">
            DATASET &nbsp;&nbsp; UCI AI4I 2020<br>
            RECORDS &nbsp;&nbsp; 10,000<br>
            MODEL &nbsp;&nbsp;&nbsp;&nbsp; RANDOM FOREST<br>
            ACCURACY &nbsp; 90.2%<br>
            FEATURES &nbsp; 9 PARAMS<br>
            STATUS &nbsp;&nbsp;&nbsp; ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(air_t, proc_t, rpm, torq, wear, type_enc):
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
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
    return df

X = build_features(air_temp, process_temp, rot_speed, torque, tool_wear, type_encoded)

if model is not None:
    prob      = float(model.predict_proba(X)[0][1])
    predicted = int(model.predict(X)[0])
else:
    rs = (
        (torque  - 3.8)  / (76.6  - 3.8)  * 0.35 +
        (tool_wear / 253)                  * 0.40 +
        ((rot_speed - 1168) / (2886-1168)) * 0.15 +
        (max(0, process_temp - air_temp - 8.6) / 10) * 0.10
    )
    prob      = float(np.clip(rs, 0.02, 0.97))
    predicted = 1 if prob > 0.5 else 0


# ── Gauge ─────────────────────────────────────────────────────────────────────
def draw_gauge(prob):
    fig = plt.figure(figsize=(5, 3.3), facecolor='none')
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('#08090c')
    fig.patch.set_facecolor('#08090c')

    START, SPAN = np.pi * 1.12, np.pi * 1.24

    # Background track
    th_bg = np.linspace(START, START - SPAN, 300)
    ax.plot(th_bg, [1]*300, color='#141e2c', linewidth=24, solid_capstyle='round', zorder=1)

    # Zone shading
    zones = [(0,.33,'#00cc44',.18),(0.33,.66,'#ff8c00',.18),(0.66,1,'#ff3a1a',.18)]
    for zs,ze,zc,za in zones:
        zt = np.linspace(START - zs*SPAN, START - ze*SPAN, 100)
        ax.plot(zt,[1]*100,color=zc,linewidth=24,solid_capstyle='round',alpha=za,zorder=2)

    # Fill arc
    fill_angle = START - prob * SPAN
    tf = np.linspace(START, fill_angle, 300)
    fc = '#00cc44' if prob<.33 else ('#ff8c00' if prob<.66 else '#ff3a1a')
    ax.plot(tf,[1]*300,color=fc,linewidth=24,solid_capstyle='round',zorder=3)
    ax.plot(tf,[1]*300,color=fc,linewidth=34,solid_capstyle='round',alpha=.09,zorder=2)

    # Tick marks
    for i in range(11):
        a   = START - (i/10)*SPAN
        rin = .82 if i%5==0 else .87
        ax.plot([a,a],[rin,.93],color='#1e3050',linewidth=1.8 if i%5==0 else 1,zorder=4)
        if i%5==0:
            ax.text(a,.72,f'{i*10}%',ha='center',va='center',
                    color='#2a4060',fontsize=8,fontfamily='monospace')

    # Needle
    needle_a = START - prob*SPAN
    ax.annotate('',xy=(needle_a,.9),xytext=(needle_a+np.pi,.1),
                arrowprops=dict(arrowstyle='-|>',color='#e8eef2',lw=2.2,mutation_scale=12))
    ax.plot(0,0,'o',color='#e8eef2',markersize=7,zorder=6)
    ax.plot(0,0,'o',color='#08090c',markersize=4,zorder=7)

    # Centre text
    ax.text(0,.05,f'{prob*100:.1f}%',ha='center',va='center',
            color=fc,fontsize=28,fontweight='bold',fontfamily='monospace',zorder=8)
    ax.text(0,-.28,'FAILURE RISK',ha='center',va='center',
            color='#2a4060',fontsize=9,fontfamily='monospace')

    ax.set_ylim(0,1.45)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ── Feature importance ────────────────────────────────────────────────────────
def draw_importance():
    if model is None: return None
    try:
        imp    = model.feature_importances_
        labels = list(feature_names) if feature_names is not None else [f"F{i}" for i in range(len(imp))]
        idx    = np.argsort(imp)[-8:]

        fig, ax = plt.subplots(figsize=(5,3.6))
        fig.patch.set_facecolor('#08090c')
        ax.set_facecolor('#08090c')

        vals   = [imp[i] for i in idx]
        names  = [labels[i].replace(' [','\n[') for i in idx]
        colors = ['#ff3a1a' if v>.25 else ('#ff8c00' if v>.12 else '#ffb347') for v in vals]

        ax.barh(range(len(idx)),vals,color=colors,edgecolor='none',height=0.55,alpha=.9)
        ax.barh(range(len(idx)),vals,color=colors,edgecolor='none',height=0.55,alpha=.12)

        for i,(v,c) in enumerate(zip(vals,colors)):
            ax.text(v+.003,i,f'{v:.3f}',va='center',color='#3a5570',fontsize=9,fontfamily='monospace')

        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(names,fontsize=8,color='#5a7a95',fontfamily='monospace')
        ax.set_xlabel('Importance Score',color='#2a4060',fontsize=9,fontfamily='monospace')
        ax.tick_params(axis='x',colors='#2a4060',labelsize=8)
        ax.set_xlim(0,max(vals)*1.3)
        ax.xaxis.grid(True,color='#111c28',linestyle='--',linewidth=.5,alpha=.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_edgecolor('#141e2c')
        plt.tight_layout(pad=.4)
        return fig
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN UI ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Header
hl, hr = st.columns([3,1])
with hl:
    st.markdown("""
    <div style="padding:4px 0 0;">
        <div class="hud-title">⚙ MachineGuard AI</div>
        <div class="hud-sub">
            <span class="live-dot"></span>
            REAL-TIME PREDICTIVE MAINTENANCE &nbsp;|&nbsp; UCI AI4I 2020 DATASET
        </div>
    </div>
    """, unsafe_allow_html=True)
with hr:
    status_color = "#ff4500" if predicted else "#00cc44"
    status_text  = "FAILURE DETECTED" if predicted else "ALL NOMINAL"
    st.markdown(f"""
    <div style="text-align:right;padding-top:6px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2a4060;letter-spacing:.15em;">
            SYSTEM STATUS
        </div>
        <div style="font-family:'Orbitron',monospace;font-size:12px;
                    color:{status_color};letter-spacing:.08em;
                    text-shadow:0 0 12px {status_color};">
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠ Model file not found — running in DEMO MODE.", icon="⚠️")

# KPI Row
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("FAILURE PROB",    f"{prob*100:.1f}%")
k2.metric("SYSTEM STATUS",   "FAILURE" if predicted else "NORMAL")
k3.metric("HEALTH INDEX",    f"{max(0,100-int(prob*100))}%")
k4.metric("TOOL WEAR",       f"{tool_wear} / 253 min")
k5.metric("TEMP DELTA",      f"{process_temp-air_temp:.1f} K")

st.markdown("<br>", unsafe_allow_html=True)

# Status Banner
if predicted or prob > 0.5:
    st.markdown(f"""
    <div class="banner-alert">
        <div class="banner-alert-title">🚨 CRITICAL FAILURE IMMINENT</div>
        <div class="banner-alert-sub">
            Failure probability: <strong>{prob*100:.1f}%</strong> &nbsp;—&nbsp;
            Immediate maintenance required. Isolate machine and dispatch technician now.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="banner-safe">
        <div class="banner-safe-title">✅ ALL SYSTEMS NOMINAL</div>
        <div class="banner-safe-sub">
            Failure probability: <strong>{prob*100:.1f}%</strong> &nbsp;—&nbsp;
            All sensors within operational tolerance. Standard monitoring active.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Gauge | Feature Importance | Risk Bars
cg, cf, cr = st.columns([1.1, 1.2, 0.9])

with cg:
    st.markdown('<div class="sec-label">RISK GAUGE</div>', unsafe_allow_html=True)
    gfig = draw_gauge(prob)
    st.pyplot(gfig, use_container_width=True)
    plt.close(gfig)

with cf:
    st.markdown('<div class="sec-label">FEATURE IMPORTANCE RANKING</div>', unsafe_allow_html=True)
    ifig = draw_importance()
    if ifig:
        st.pyplot(ifig, use_container_width=True)
        plt.close(ifig)
    else:
        st.info("Loads with trained model.")

with cr:
    st.markdown('<div class="sec-label">RISK FACTOR BREAKDOWN</div>', unsafe_allow_html=True)
    factors = [
        ("TOOL WEAR",      tool_wear/253,                                "#ff3a1a"),
        ("TORQUE STRESS",  max(0,(torque-40)/36.6),                     "#ff8c00"),
        ("TEMP DELTA",     max(0,(process_temp-air_temp-8.6)/5),        "#ffb347"),
        ("SPEED ANOMALY",  max(0,abs(rot_speed-1800)/1386),             "#ffd580"),
        ("PROCESS HEAT",   max(0,(process_temp-308)/7),                 "#ff8c00"),
    ]
    for name, val, color in factors:
        pct = min(100, int(val*100))
        st.markdown(f"""
        <div class="riskbar-wrap">
            <div class="riskbar-head">
                <span>{name}</span>
                <span style="color:{color};font-weight:bold;">{pct}%</span>
            </div>
            <div class="riskbar-track">
                <div class="riskbar-fill"
                     style="width:{pct}%;background:linear-gradient(90deg,{color}55,{color});">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)

# Sensor Telemetry Table
st.markdown('<div class="sec-label">LIVE SENSOR TELEMETRY</div>', unsafe_allow_html=True)

def sstatus(v, lo, hi, crit=None):
    if crit and v > crit: return "🔴 CRITICAL"
    if v < lo or v > hi:  return "🟡 WARNING"
    return "🟢 NOMINAL"

sensor_df = pd.DataFrame({
    "PARAMETER":      ["Air Temp","Process Temp","Temp Differential","Rotational Speed","Torque","Tool Wear","Machine Type"],
    "READING":        [f"{air_temp:.1f} K", f"{process_temp:.1f} K",
                       f"{process_temp-air_temp:.1f} K", f"{rot_speed} rpm",
                       f"{torque:.1f} Nm", f"{tool_wear} min", machine_type.split('—')[0].strip()],
    "NOMINAL RANGE":  ["295–304 K","305–314 K","8–12 K","1300–2500 rpm","≤55 Nm","≤200 min","L/M/H"],
    "STATUS":         [
        sstatus(air_temp,295,304),
        sstatus(process_temp,305,314),
        sstatus(process_temp-air_temp,8,12),
        sstatus(rot_speed,1300,2500),
        sstatus(torque,0,55,crit=70),
        sstatus(tool_wear,0,200,crit=240),
        "🟢 NOMINAL",
    ],
})
st.dataframe(sensor_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Analysis charts from repo
chart_pairs = [
    ("feature_importance.png", "FEATURE IMPORTANCE ANALYSIS"),
    ("correlation_heatmap.png", "SENSOR CORRELATION MATRIX"),
]
available = [(f,l) for f,l in chart_pairs if os.path.exists(f)]
if available:
    st.markdown('<div class="sec-label">MODEL ANALYSIS CHARTS</div>', unsafe_allow_html=True)
    icols = st.columns(len(available))
    for col,(fname,label) in zip(icols, available):
        with col:
            st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#2a4060;margin-bottom:6px;letter-spacing:.1em;">{label}</div>', unsafe_allow_html=True)
            st.image(fname, use_container_width=True)

if os.path.exists("sensor_histograms.png"):
    st.markdown('<div class="sec-label" style="margin-top:14px;">SENSOR DISTRIBUTION PROFILES</div>', unsafe_allow_html=True)
    st.image("sensor_histograms.png", use_container_width=True)

# Footer
st.markdown('<div class="hud-divider" style="margin-top:28px;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;padding:6px 0 16px;
            font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.08em;">
    <span style="color:#1a2d40;">
        MACHINEGUARD AI &nbsp;|&nbsp; RANDOM FOREST CLASSIFIER &nbsp;|&nbsp; 90.2% ACCURACY &nbsp;|&nbsp; UCI AI4I 2020
    </span>
    <span>
        <a href="https://github.com/Sneh-04/Machine_Failure_Prediction"
           style="color:#ff8c00;text-decoration:none;">⚙ GITHUB</a>
        &nbsp;|&nbsp;
        <a href="https://linkedin.com/in/sneha-kunduru"
           style="color:#3a5570;text-decoration:none;">LINKEDIN</a>
    </span>
</div>
""", unsafe_allow_html=True)
