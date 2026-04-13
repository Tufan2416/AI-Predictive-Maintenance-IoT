# ============================================================
# dashboard/app.py — Streamlit Live Monitoring Dashboard
# ============================================================
# Run: streamlit run dashboard/app.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
from datetime import datetime

from src.utils      import load_config, classify_alert, get_maintenance_recommendation
from src.preprocess import engineer_features, get_feature_columns

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #fafafa; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 6px 0;
        border-left: 4px solid #2196F3;
    }
    .alert-normal   { border-left-color: #4CAF50 !important; }
    .alert-warning  { border-left-color: #FFC107 !important; }
    .alert-critical { border-left-color: #F44336 !important; }
    .big-metric { font-size: 2.2rem; font-weight: 700; }
    .stProgress > div > div { background-color: #2196F3; }
    div[data-testid="stSidebarContent"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources ────────────────────────────────────────────
@st.cache_resource
def load_resources():
    config = load_config()
    model_path  = config["paths"]["model_output"]
    scaler_path = config["paths"]["scaler_output"]

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, config, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    proc_path = config["paths"]["processed_data"]
    df = pd.read_csv(proc_path, parse_dates=["timestamp"]) \
        if os.path.exists(proc_path) else None

    feat_cols = get_feature_columns(df) if df is not None else []
    return model, scaler, config, df, feat_cols


# ── Simulate Live Reading ─────────────────────────────────────
def get_live_reading(mode: str = "normal") -> dict:
    """Generate one simulated sensor reading."""
    np.random.seed(int(time.time()) % 10000)
    if mode == "normal":
        return {
            "temperature":  round(np.random.normal(65, 5), 2),
            "vibration":    round(np.random.normal(2.0, 0.3), 3),
            "current":      round(np.random.normal(12, 1.5), 2),
            "pressure":     round(np.random.normal(65, 5), 2),
            "runtime_hours": round(np.random.uniform(100, 700), 1)
        }
    elif mode == "warning":
        return {
            "temperature":  round(np.random.normal(85, 5), 2),
            "vibration":    round(np.random.normal(4.5, 0.5), 3),
            "current":      round(np.random.normal(18, 2), 2),
            "pressure":     round(np.random.normal(90, 6), 2),
            "runtime_hours": round(np.random.uniform(800, 950), 1)
        }
    else:   # critical
        return {
            "temperature":  round(np.random.normal(105, 8), 2),
            "vibration":    round(np.random.normal(7.5, 0.8), 3),
            "current":      round(np.random.normal(23, 2), 2),
            "pressure":     round(np.random.normal(112, 8), 2),
            "runtime_hours": round(np.random.uniform(950, 1100), 1)
        }


# ── Predict from Reading ──────────────────────────────────────
def predict_reading(reading: dict, model, scaler, feat_cols: list, config: dict) -> dict:
    row_df = pd.DataFrame([reading])
    row_df["timestamp"] = pd.Timestamp.now()
    row_df["failure"]   = 0
    padded = pd.concat([row_df] * 12, ignore_index=True)
    padded = engineer_features(padded, window=10)
    row_feat = padded.iloc[[-1]][feat_cols].values
    row_scaled = scaler.transform(row_feat)
    prob = model.predict_proba(row_scaled)[0][1]
    pred = int(prob >= 0.5)
    alert = classify_alert(reading, config["thresholds"])
    return {"prediction": pred, "failure_prob": round(prob, 4), "alert_level": alert}


# ── Alert Badge HTML ──────────────────────────────────────────
def alert_badge(level: str) -> str:
    colors = {"NORMAL": "#4CAF50", "WARNING": "#FFC107", "CRITICAL": "#F44336"}
    icons  = {"NORMAL": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}
    c = colors.get(level, "#888")
    i = icons.get(level, "❓")
    return (
        f'<div style="background:{c};color:#000;padding:10px 20px;'
        f'border-radius:8px;font-size:1.4rem;font-weight:700;'
        f'text-align:center;">{i} {level}</div>'
    )


# ── Main App ──────────────────────────────────────────────────
def main():
    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/robot.png", width=60)
        st.title("⚙️ PredictMaint AI")
        st.markdown("---")

        mode = st.selectbox(
            "🎛️ Simulate Machine State",
            ["normal", "warning", "critical"],
            index=0,
            help="Simulates different machine health states"
        )

        live_mode = st.toggle("🔴 Live Simulation Mode", value=False)
        refresh   = st.slider("Refresh interval (s)", 1, 10, 3)

        st.markdown("---")
        st.markdown("### 📊 Thresholds")
        resources = load_resources()
        if len(resources) == 5:
            _, _, config, _, _ = resources
            t = config["thresholds"]
            for sensor, vals in t.items():
                if sensor in ["temperature", "vibration", "current", "pressure"]:
                    st.caption(
                        f"**{sensor.capitalize()}**: "
                        f"Warn >{vals['warning_max']} | "
                        f"Critical >{vals['critical_max']}"
                    )
        st.markdown("---")
        st.caption("AI Predictive Maintenance v1.0")
        st.caption("Built with Python · Scikit-learn · Streamlit")

    # ── Check resources ────────────────────────────────────────
    resources = load_resources()
    if len(resources) != 5 or resources[0] is None:
        st.error(
            "🚫 Model not found! Please run `python main.py --train` first, "
            "then refresh this page."
        )
        st.code("python main.py --train", language="bash")
        return

    model, scaler, config, hist_df, feat_cols = resources

    # ── Title Row ──────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#2196F3;margin-bottom:0'>⚙️ Predictive Maintenance Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.caption("Real-time industrial IoT sensor monitoring & AI failure prediction")
    st.markdown("---")

    # ── Session State for live history ────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── Get current reading ────────────────────────────────────
    reading = get_live_reading(mode)
    result  = predict_reading(reading, model, scaler, feat_cols, config)
    alert   = result["alert_level"]
    prob    = result["failure_prob"]

    # Add to history
    st.session_state.history.append({
        **reading,
        "timestamp":    datetime.now().strftime("%H:%M:%S"),
        "failure_prob": prob,
        "alert_level":  alert
    })
    history_window = config["dashboard"]["history_window"]
    if len(st.session_state.history) > history_window:
        st.session_state.history = st.session_state.history[-history_window:]

    hist = pd.DataFrame(st.session_state.history)

    # ── KPI Row ────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🌡️ Temperature (°C)",  f"{reading['temperature']:.1f}",
                delta_color="inverse")
    col2.metric("📳 Vibration (mm/s)",  f"{reading['vibration']:.2f}")
    col3.metric("⚡ Current (A)",       f"{reading['current']:.1f}")
    col4.metric("🔧 Pressure (PSI)",    f"{reading['pressure']:.1f}")
    col5.metric("⏱️ Runtime (hrs)",     f"{reading['runtime_hours']:.0f}")

    st.markdown("---")

    # ── Alert + Failure Probability ───────────────────────────
    a1, a2, a3 = st.columns([1, 1.5, 2])

    with a1:
        st.markdown("#### Alert Status")
        st.markdown(alert_badge(alert), unsafe_allow_html=True)

    with a2:
        st.markdown("#### Failure Probability")
        colour = "#4CAF50" if prob < 0.35 else "#FFC107" if prob < 0.65 else "#F44336"
        st.markdown(
            f'<div class="big-metric" style="color:{colour}">{prob:.1%}</div>',
            unsafe_allow_html=True
        )
        st.progress(min(prob, 1.0))

    with a3:
        st.markdown("#### Maintenance Recommendation")
        top_feats = ["vibration", "temperature", "current"]
        rec = get_maintenance_recommendation(alert, prob, top_feats)
        bg = {"NORMAL": "#1a3a1a", "WARNING": "#3a2a00", "CRITICAL": "#3a0a0a"}
        st.markdown(
            f'<div style="background:{bg.get(alert,"#1a1a1a")};padding:14px;'
            f'border-radius:8px;font-size:0.9rem">{rec}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Live Charts ────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📈 Sensor Trend (Live Window)")
        if len(hist) > 1:
            fig = go.Figure()
            for sensor, color in [
                ("temperature", "#FF7043"),
                ("vibration",   "#42A5F5"),
                ("current",     "#66BB6A"),
                ("pressure",    "#AB47BC")
            ]:
                if sensor in hist.columns:
                    fig.add_trace(go.Scatter(
                        y=hist[sensor], name=sensor.capitalize(),
                        line=dict(color=color, width=2),
                        mode="lines"
                    ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🚨 Failure Probability Over Time")
        if len(hist) > 1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=hist["failure_prob"],
                fill="tozeroy",
                line=dict(color="#EF5350", width=2),
                name="Failure Prob"
            ))
            fig2.add_hline(y=0.5, line_dash="dash",
                           line_color="yellow",
                           annotation_text="Decision Threshold (0.5)")
            fig2.update_layout(
                template="plotly_dark", height=300,
                yaxis_range=[0, 1],
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Historical Analysis (from full dataset) ───────────────
    if hist_df is not None and len(hist_df) > 0:
        st.markdown("---")
        st.markdown("#### 🗂️ Historical Sensor Analysis (Training Dataset)")
        tab1, tab2, tab3 = st.tabs([
            "Sensor Distributions", "Failure Events", "Correlation"
        ])

        with tab1:
            sensor = st.selectbox(
                "Select sensor", ["temperature", "vibration", "current", "pressure"]
            )
            fig3 = px.histogram(
                hist_df, x=sensor, color="failure",
                color_discrete_map={0: "#1E88E5", 1: "#E53935"},
                labels={"failure": "Failure"},
                nbins=50, barmode="overlay", opacity=0.75,
                title=f"{sensor.capitalize()} Distribution — Normal vs Failure"
            )
            fig3.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            sample = hist_df.sample(min(1000, len(hist_df)),
                                     random_state=42).reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=sample.index, y=sample["temperature"],
                mode="lines", name="Temperature",
                line=dict(color="#42A5F5", width=0.8)
            ))
            failure_pts = sample[sample["failure"] == 1]
            fig4.add_trace(go.Scatter(
                x=failure_pts.index, y=failure_pts["temperature"],
                mode="markers", name="Failure Event",
                marker=dict(color="#E53935", size=6, symbol="x")
            ))
            fig4.update_layout(
                template="plotly_dark", height=350,
                title="Temperature Signal with Failure Events"
            )
            st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            corr_cols = ["temperature", "vibration", "current", "pressure",
                         "stress_score", "failure"]
            corr_cols = [c for c in corr_cols if c in hist_df.columns]
            corr = hist_df[corr_cols].corr()
            fig5 = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Feature Correlation Heatmap"
            )
            fig5.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig5, use_container_width=True)

    # ── Live auto-refresh ─────────────────────────────────────
    if live_mode:
        time.sleep(refresh)
        st.rerun()


if __name__ == "__main__":
    main()
