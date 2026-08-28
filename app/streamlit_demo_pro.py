"""
Sanjeevani AI — Live Demo of Classical vs Quantum Models for Heart Disease Risk Prediction
This Streamlit app allows you to input patient data and see how three different models predict the risk of heart disease: a classical SVM, a quantum kernel SVM (QSVC), and a variational quantum classifier (VQC). The app displays the predictions as color-coded speedometer gauges, along with inference times and model agreement.
The app also includes a comparison chart of the models' performance metrics (accuracy and F1 score)
=============================================
Shows THREE models side by side: Classical SVM, Quantum Kernel SVM
(QSVC), and VQC — each as a color-coded speedometer gauge.
"""

"""
Sanjeevani AI — Live Demo (Deliverable 3)
=============================================
Shows THREE models side by side: Classical SVM, Quantum Kernel SVM
(QSVC), and VQC — each as a color-coded speedometer gauge. Includes
one-click example patients for a fast, reliable live demo.

Run from your project root:
    streamlit run app/streamlit_demo_pro.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.algorithms import VQC

st.set_page_config(
    page_title="Sanjeevani AI | Heart Risk Screening",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 46px; font-weight: 800;
    background: linear-gradient(90deg, #0B5E5B, #14A085);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0px;
}
.subtitle {
    text-align: center; color: #999; font-size: 18px;
    margin-top: 4px; margin-bottom: 14px;
}
div[data-testid="stMetricValue"] { font-size: 22px; }
.section-label {
    font-size: 20px; font-weight: 700; margin-top: 10px; margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🫀 Sanjeevani AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Detecting risk before it becomes an emergency '
    '&nbsp;·&nbsp; Classical AI vs Quantum Kernel SVM vs VQC</p>',
    unsafe_allow_html=True
)
st.markdown("---")

@st.cache_resource
def load_all():
    classical_full = joblib.load("models/classical_full_model.joblib")
    quantum_model = joblib.load("models/quantum_qsvc_model.joblib")
    quantum_scaler = joblib.load("models/quantum_angle_scaler.joblib")
    vqc_model = VQC.from_dill("models/vqc_model.model")
    vqc_scaler = joblib.load("models/vqc_angle_scaler.joblib")
    data = np.load("data/prepared_data.npz", allow_pickle=True)
    raw_df = pd.read_csv("data/heart.csv", encoding="utf-8-sig")
    return classical_full, quantum_model, quantum_scaler, vqc_model, vqc_scaler, data, raw_df

try:
    (classical_full, quantum_model, quantum_scaler,
     vqc_model, vqc_scaler, data, raw_df) = load_all()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load one or more models: {e}")
    st.info("Make sure you've run 02_classical_model.py, 03_quantum_model.py, "
            "and 04_vqc_model.py first — each one saves the files this demo needs.")
    models_loaded = False

if models_loaded:
    feature_names = list(data["feature_names"])
    top4_features = list(data["top4_features"])
    top4_idx = [feature_names.index(f) for f in top4_features]

    feature_display_names = {
        "age": "Age (years)", "sex": "Sex", "cp": "Chest Pain Type",
        "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Sugar >120", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "Major Vessels", "thal": "Thalassemia",
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

    st.markdown('<p class="section-label">⚡ Quick Demo Examples</p>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        if st.button("🔴 Load High-Risk Example", use_container_width=True):
            row = raw_df[raw_df["target"] == 1].sample(1, random_state=1).iloc[0]
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(row[feat])
            st.rerun()

    with ex2:
        if st.button("🟢 Load Low-Risk Example", use_container_width=True):
            row = raw_df[raw_df["target"] == 0].sample(1, random_state=2).iloc[0]
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(row[feat])
            st.rerun()

    with ex3:
        if st.button("↺ Reset to Median", use_container_width=True):
            for feat in feature_names:
                st.session_state[f"input_{feat}"] = float(raw_df[feat].median())
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-label">📋 Patient Details</p>', unsafe_allow_html=True)

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

    st.markdown("---")
    predict_clicked = st.button("🔍 RUN PREDICTION", type="primary", use_container_width=True)

    def safe_predict(model, X):
        result = np.atleast_1d(model.predict(X))
        return int(result.flatten()[0])

    def safe_predict_proba_positive(model, X):
        result = np.atleast_2d(model.predict_proba(X))
        return float(result[0][1])

    def make_gauge(value_pct, title, subtitle_note=""):
        if value_pct < 35:
            bar_color, zone_text = "#2ecc71", "Low Risk"
        elif value_pct < 65:
            bar_color, zone_text = "#f39c12", "Moderate Risk"
        else:
            bar_color, zone_text = "#e74c3c", "High Risk"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value_pct,
            title={"text": f"{title}<br>"
                            f"<span style='font-size:14px;color:{bar_color}'>{zone_text}</span>"
                            f"<br><span style='font-size:11px;color:#999'>{subtitle_note}</span>"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 35], "color": "#eafaf1"},
                    {"range": [35, 65], "color": "#fef5e7"},
                    {"range": [65, 100], "color": "#fdedec"},
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=70, b=10))
        return fig

    if predict_clicked:
        with st.spinner("Running classical model and quantum circuit simulations..."):
            raw_vector = np.array([[inputs[f] for f in feature_names]])
            scaler = StandardScaler().fit(raw_df[feature_names])
            scaled_vector = scaler.transform(raw_vector)
            scaled_4 = scaled_vector[:, top4_idx]

            t0 = time.time()
            classical_pred = classical_full.predict(scaled_vector)[0]
            classical_proba = classical_full.predict_proba(scaled_vector)[0][1] * 100
            classical_time = (time.time() - t0) * 1000

            quantum_input = quantum_scaler.transform(scaled_4)
            t0 = time.time()
            quantum_pred = quantum_model.predict(quantum_input)[0]
            quantum_time = (time.time() - t0) * 1000
            try:
                quantum_score = quantum_model.decision_function(quantum_input)[0]
                quantum_proba = 100 / (1 + np.exp(-quantum_score))
            except Exception:
                quantum_proba = 100.0 if quantum_pred == 1 else 0.0

            vqc_input = vqc_scaler.transform(scaled_4)
            t0 = time.time()
            vqc_pred = safe_predict(vqc_model, vqc_input)
            vqc_time = (time.time() - t0) * 1000
            try:
                vqc_proba = safe_predict_proba_positive(vqc_model, vqc_input) * 100
            except Exception:
                vqc_proba = 100.0 if vqc_pred == 1 else 0.0

        st.markdown("## 🎯 Results")
        g1, g2, g3 = st.columns(3)

        with g1:
            st.plotly_chart(make_gauge(classical_proba, "🖥️ Classical AI",
                                        "SVM · 13 features"), use_container_width=True)
            st.caption(f"Inference: {classical_time:.1f} ms")

        with g2:
            st.plotly_chart(make_gauge(quantum_proba, "⚛️ Quantum Kernel SVM",
                                        f"{len(top4_features)} qubits"), use_container_width=True)
            st.caption(f"Features: {', '.join(top4_features)} · Inference: {quantum_time:.1f} ms")

        with g3:
            st.plotly_chart(make_gauge(vqc_proba, "🌀 VQC",
                                        f"{len(top4_features)} qubits"), use_container_width=True)
            st.caption(f"Features: {', '.join(top4_features)} · Inference: {vqc_time:.1f} ms")

        preds = {"Classical": classical_pred, "QSVC": quantum_pred, "VQC": vqc_pred}
        agree_count = len(set(preds.values()))
        if agree_count == 1:
            st.success("✅ All three models **AGREE** on this prediction.")
        else:
            st.warning(f"⚠️ Models **disagree** — Classical: "
                       f"{'At risk' if classical_pred else 'Lower risk'}, "
                       f"QSVC: {'At risk' if quantum_pred else 'Lower risk'}, "
                       f"VQC: {'At risk' if vqc_pred else 'Lower risk'}. "
                       f"An interesting case where the models' decision boundaries diverge.")

        if any(p == 1 for p in preds.values()):
            st.info("💡 In a real deployment, this patient would be flagged for clinical follow-up.")

    st.markdown("---")
    with st.expander("📊 Full Model Comparison (Deliverable 4)"):
        try:
            st.image("results/comparison_chart.png", use_container_width=True)
        except Exception:
            st.write("Run 05_comparison.py first to generate the chart.")
        try:
            with open("results/comparison_summary.md") as f:
                st.markdown(f.read())
        except Exception:
            pass

    st.markdown("---")
    st.caption("Sanjeevani AI · IEEE Region 10 QAI-Lead 2026 · "
               "Built with Qiskit + scikit-learn + Streamlit")