"""
Sanjeevani AI — Live Demo of Classical vs Quantum Models for Heart Disease Risk Prediction

A polished Streamlit dashboard comparing:
    • Classical SVM
    • Quantum Kernel SVM (QSVC)
    • Variational Quantum Classifier (VQC)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from qiskit_machine_learning.algorithms import VQC


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sanjeevani AI | Heart Risk Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# VISUAL THEME
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --heart: #b4233c;
            --heart-dark: #7f1730;
            --heart-soft: #fff1f3;
            --teal: #0f8b8d;
            --teal-dark: #0a6264;
            --teal-soft: #edfafa;
            --ink: #17202a;
            --muted: #667085;
            --border: #eadcdf;
            --card: #ffffff;
            --canvas: #fffafb;
            --shadow: 0 10px 30px rgba(98, 35, 52, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 4%, rgba(180, 35, 60, 0.07), transparent 24%),
                radial-gradient(circle at 92% 7%, rgba(15, 139, 141, 0.07), transparent 22%),
                linear-gradient(180deg, #fffafb 0%, #ffffff 35%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.3rem;
            padding-bottom: 2.5rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Header */
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(180, 35, 60, 0.13);
            border-radius: 26px;
            padding: 1.6rem 1.8rem 1.45rem 1.8rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffffff 0%, #fff4f6 58%, #f4fbfb 100%);
            box-shadow: var(--shadow);
        }

        .hero::after {
            content: "";
            position: absolute;
            right: -70px;
            top: -90px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: rgba(180, 35, 60, 0.05);
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 18px;
            position: relative;
            z-index: 1;
        }

        .brand-title {
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 3.6rem);
            line-height: 1;
            font-weight: 850;
            letter-spacing: -0.04em;
            color: var(--heart-dark);
        }

        .brand-kicker {
            margin: 6px 0 0 0;
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--teal-dark);
        }

        .brand-subtitle {
            margin: 7px 0 0 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.5;
            max-width: 820px;
        }

        .pulse {
            margin-top: 17px;
            height: 3px;
            width: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--heart), var(--teal), transparent 85%);
        }

        /* Section headings */
        .section-head {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 1.15rem 0 0.7rem 0;
        }

        .section-head .icon {
            width: 36px;
            height: 36px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: var(--heart-soft);
            border: 1px solid rgba(180, 35, 60, 0.11);
            font-size: 1.1rem;
        }

        .section-head h2 {
            margin: 0;
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--ink);
        }

        .section-head p {
            margin: 2px 0 0 0;
            color: var(--muted);
            font-size: 0.86rem;
        }

        /* Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
        }

        .info-strip {
            display: flex;
            align-items: center;
            gap: 12px;
            border-radius: 16px;
            padding: 12px 14px;
            margin: 5px 0 14px 0;
            background: linear-gradient(90deg, #fff5f6, #f5fbfb);
            border: 1px solid var(--border);
            color: #475467;
            font-size: 0.88rem;
        }

        .info-strip strong {
            color: var(--heart-dark);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 13px;
            min-height: 44px;
            font-weight: 750;
            border: 1px solid #e4d5d9;
            background: rgba(255, 255, 255, 0.92);
            color: var(--ink);
            box-shadow: 0 4px 14px rgba(64, 30, 39, 0.05);
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            border-color: #cf8492;
            color: var(--heart-dark);
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(180, 35, 60, 0.10);
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--heart), var(--heart-dark));
            border: none;
            color: white;
            box-shadow: 0 10px 24px rgba(180, 35, 60, 0.20);
        }

        div.stButton > button[kind="primary"]:hover {
            color: white;
            box-shadow: 0 14px 28px rgba(180, 35, 60, 0.26);
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 12px;
            border-color: #e6dadd;
        }

        div[data-testid="stSlider"] {
            padding: 1px 2px 7px 2px;
        }

        label p {
            font-weight: 650 !important;
            color: #344054 !important;
        }

        /* Prediction cards */
        .model-card {
            min-height: 92px;
            padding: 15px 16px;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(49, 28, 34, 0.055);
            margin-bottom: 10px;
        }

        .model-card.classical {
            border-top: 4px solid #b4233c;
        }

        .model-card.quantum {
            border-top: 4px solid #0f8b8d;
        }

        .model-card.vqc {
            border-top: 4px solid #7057c8;
        }

        .model-card .name {
            font-weight: 800;
            color: #1d2939;
            font-size: 0.97rem;
        }

        .model-card .meta {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 3px;
        }

        /* Plotly containers */
        div[data-testid="stPlotlyChart"] {
            border-radius: 18px;
            background: rgba(255,255,255,0.75);
            border: 1px solid #f0e5e8;
            padding: 3px;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 15px;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            border-radius: 18px;
            border-color: var(--border);
            background: rgba(255,255,255,0.78);
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #98a2b3;
            font-size: 0.78rem;
            padding: 1.2rem 0 0.2rem 0;
        }

        /* Hide Streamlit chrome that adds visual noise */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HEADER / LOGO
# -----------------------------------------------------------------------------
logo_path = os.path.join("assets", "sanjeevani logo.png")
logo_html = ""
if os.path.exists(logo_path):
    # Streamlit image is used below; HTML contains only the textual brand shell.
    pass

hero_left, hero_right = st.columns([1, 6], vertical_alignment="center")
with hero_left:
    if os.path.exists(logo_path):
        st.image(logo_path, width=112)
    else:
        st.markdown("<div style='font-size:5rem;line-height:1'>🫀</div>", unsafe_allow_html=True)

with hero_right:
    st.markdown(
        """
        <div class="hero">
            <div class="brand-row">
                <div>
                    <p class="brand-kicker">AI-powered cardiac risk screening</p>
                    <h1 class="brand-title">Sanjeevani AI</h1>
                    <p class="brand-subtitle">
                        Detecting risk before it becomes an emergency ·
                        Classical AI vs Quantum Kernel SVM vs VQC
                    </p>
                    <div class="pulse"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MODEL LOADING — SAVED SCALER INCLUDED
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all():
    classical_full = joblib.load("models/classical_full_model.joblib")
    quantum_model = joblib.load("models/quantum_qsvc_model.joblib")
    quantum_scaler = joblib.load("models/quantum_angle_scaler.joblib")
    vqc_model = VQC.from_dill("models/vqc_model.model")
    vqc_scaler = joblib.load("models/vqc_angle_scaler.joblib")
    feature_scaler = joblib.load("models/feature_scaler.joblib")
    data = np.load("data/prepared_data.npz", allow_pickle=True)
    raw_df = pd.read_csv("data/heart.csv", encoding="utf-8-sig")
    return (
        classical_full,
        quantum_model,
        quantum_scaler,
        vqc_model,
        vqc_scaler,
        feature_scaler,
        data,
        raw_df,
    )


try:
    (
        classical_full,
        quantum_model,
        quantum_scaler,
        vqc_model,
        vqc_scaler,
        feature_scaler,
        data,
        raw_df,
    ) = load_all()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load one or more models: {e}")
    st.info(
        "Make sure you've run 02_classical_model.py, 03_quantum_model.py, "
        "and 04_vqc_model.py first — each one saves the files this demo needs."
    )
    models_loaded = False


if models_loaded:
    feature_names = list(data["feature_names"])
    top4_features = list(data["top4_features"])
    top4_idx = [feature_names.index(f) for f in top4_features]

    feature_display_names = {
        "age": "Age (years)",
        "sex": "Sex",
        "cp": "Chest Pain Type",
        "trestbps": "Resting BP (mm Hg)",
        "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Sugar >120",
        "restecg": "Resting ECG",
        "thalach": "Max Heart Rate",
        "exang": "Exercise Angina",
        "oldpeak": "ST Depression",
        "slope": "ST Slope",
        "ca": "Major Vessels",
        "thal": "Thalassemia",
    }

    def build_label(feat):
        base = feature_display_names.get(feat, feat)
        if raw_df[feat].nunique() <= 4:
            vals = sorted(raw_df[feat].unique().tolist())
            return f"{base} ({', '.join(str(int(v)) for v in vals)})"
        lo, hi = int(raw_df[feat].min()), int(raw_df[feat].max())
        return f"{base} ({lo}-{hi})"

    for feat in feature_names:
        key = f"input_{feat}"
        if key not in st.session_state:
            st.session_state[key] = float(raw_df[feat].median())

    # -------------------------------------------------------------------------
    # QUICK DEMO
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-head">
            <div class="icon">⚡</div>
            <div>
                <h2>Quick Demo Examples</h2>
                <p>Load representative records or restore the median profile.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        if st.button("🔴  Load High-Risk Example", use_container_width=True):
            row = raw_df[raw_df["target"] == 1].sample(1, random_state=1).iloc[0]
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(row[feat])
            st.rerun()

    with ex2:
        if st.button("🟢  Load Low-Risk Example", use_container_width=True):
            row = raw_df[raw_df["target"] == 0].sample(1, random_state=2).iloc[0]
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(row[feat])
            st.rerun()

    with ex3:
        if st.button("↺  Reset to Median", use_container_width=True):
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(raw_df[feat].median())
            st.rerun()

    st.markdown(
        "<div class='info-strip'>💡 <span><strong>Demo mode:</strong> these examples are drawn from the loaded heart dataset and are intended for model demonstration, not clinical diagnosis.</span></div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # PATIENT INPUTS
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-head">
            <div class="icon">🩺</div>
            <div>
                <h2>Patient Details</h2>
                <p>Enter the 13 features used by the classical model.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(feature_names):
        lo, hi = float(raw_df[feat].min()), float(raw_df[feat].max())
        key = f"input_{feat}"
        with cols[i % 4]:
            if raw_df[feat].nunique() <= 4:
                options = sorted(raw_df[feat].unique().tolist())
                inputs[feat] = st.selectbox(build_label(feat), options, key=key)
            else:
                inputs[feat] = st.slider(build_label(feat), lo, hi, key=key)

    st.markdown("<div style='height:7px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button(
        "🫀  RUN RISK PREDICTION",
        type="primary",
        use_container_width=True,
    )

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def safe_predict(model, X):
        result = np.atleast_1d(model.predict(X))
        return int(result.flatten()[0])

    def safe_predict_proba_positive(model, X):
        result = np.atleast_2d(model.predict_proba(X))
        return float(result[0][1])

    def make_gauge(value_pct, title, subtitle_note=""):
        if value_pct < 35:
            bar_color, zone_text, zone_bg = "#22a06b", "Low Risk", "#eefaf4"
        elif value_pct < 65:
            bar_color, zone_text, zone_bg = "#d88a15", "Moderate Risk", "#fff8ea"
        else:
            bar_color, zone_text, zone_bg = "#c9364d", "High Risk", "#fff0f2"

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value_pct,
                title={
                    "text": (
                        f"<b>{title}</b><br>"
                        f"<span style='font-size:14px;color:{bar_color}'>{zone_text}</span><br>"
                        f"<span style='font-size:11px;color:#98A2B3'>{subtitle_note}</span>"
                    )
                },
                number={
                    "suffix": "%",
                    "font": {"size": 34, "color": "#17202A"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#D0D5DD",
                        "tickfont": {"size": 9, "color": "#667085"},
                    },
                    "bar": {"color": bar_color, "thickness": 0.27},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#EAECF0",
                    "steps": [
                        {"range": [0, 35], "color": "#eefaf4"},
                        {"range": [35, 65], "color": "#fff8ea"},
                        {"range": [65, 100], "color": "#fff0f2"},
                    ],
                    "threshold": {
                        "line": {"color": bar_color, "width": 3},
                        "thickness": 0.76,
                        "value": value_pct,
                    },
                },
            )
        )
        fig.update_layout(
            height=305,
            margin=dict(l=18, r=18, t=64, b=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig, bar_color, zone_text, zone_bg

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    if predict_clicked:
        with st.spinner("Running classical model and quantum circuit simulations..."):
            raw_vector = np.array([[inputs[f] for f in feature_names]])
            scaled_vector = feature_scaler.transform(raw_vector)
            scaled_4 = scaled_vector[:, top4_idx]

            t0 = time.time()
            classical_pred = classical_full.predict(scaled_vector)[0]
            classical_proba = classical_full.predict_proba(scaled_vector)[0][1] * 100
            classical_time = (time.time() - t0) * 1000

            quantum_input = quantum_scaler.transform(scaled_4)
            t0 = time.time()
            quantum_pred = quantum_model.predict(quantum_input)[0]
            quantum_time = (time.time() - t0) * 1000
            quantum_proba = quantum_model.predict_proba(quantum_input)[0][1] * 100

            vqc_input = vqc_scaler.transform(scaled_4)
            t0 = time.time()
            vqc_pred = safe_predict(vqc_model, vqc_input)
            vqc_time = (time.time() - t0) * 1000
            try:
                vqc_proba = safe_predict_proba_positive(vqc_model, vqc_input) * 100
            except Exception:
                vqc_proba = 100.0 if vqc_pred == 1 else 0.0

        st.markdown(
            """
            <div class="section-head" style="margin-top:1.5rem;">
                <div class="icon">🎯</div>
                <div>
                    <h2>Prediction Results</h2>
                    <p>Probability estimates returned by the three trained models.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        g1, g2, g3 = st.columns(3, gap="medium")
        gauge_values = [
            (g1, classical_proba, "🖥️ Classical AI", "SVM · 13 features", classical_time, "classical", classical_pred),
            (g2, quantum_proba, "⚛️ Quantum Kernel SVM", f"QSVC · {len(top4_features)} qubits", quantum_time, "quantum", quantum_pred),
            (g3, vqc_proba, "🌀 Variational Quantum Classifier", f"VQC · {len(top4_features)} qubits", vqc_time, "vqc", vqc_pred),
        ]

        for col, probability, title, note, inference_ms, card_class, prediction in gauge_values:
            with col:
                fig, bar_color, zone_text, zone_bg = make_gauge(probability, title, note)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                label = "At risk" if prediction else "Lower risk"
                st.markdown(
                    f"""
                    <div class="model-card {card_class}">
                        <div class="name">{label}</div>
                        <div class="meta">Inference: {inference_ms:.1f} ms</div>
                        <div class="meta">Risk band: <span style='color:{bar_color};font-weight:700'>{zone_text}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        preds = {"Classical": classical_pred, "QSVC": quantum_pred, "VQC": vqc_pred}
        agree_count = len(set(preds.values()))

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if agree_count == 1:
            st.success("✅ **All three models agree** on this prediction.")
        else:
            st.warning(
                f"⚠️ **Models disagree** — Classical: "
                f"{'At risk' if classical_pred else 'Lower risk'}, "
                f"QSVC: {'At risk' if quantum_pred else 'Lower risk'}, "
                f"VQC: {'At risk' if vqc_pred else 'Lower risk'}."
            )

        if any(p == 1 for p in preds.values()):
            st.info(
                "💡 In a real deployment, a positive model result would warrant appropriate clinical follow-up."
            )

        st.caption(
            f"Quantum features used: {', '.join(top4_features)} · "
            "Probabilities shown are model outputs, not medical diagnoses."
        )

    # -------------------------------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------------------------------
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.expander("📊  Full Model Comparison · Deliverable 4"):
        try:
            st.image("results/comparison_chart.png", use_container_width=True)
        except Exception:
            st.write("Run 05_comparison.py first to generate the chart.")
        try:
            with open("results/comparison_summary.md", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            pass

    st.markdown(
        """
        <div class="footer">
            Sanjeevani AI · IEEE Region 10 QAI-Lead 2026 · Built with Qiskit + scikit-learn + Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
