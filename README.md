<div align="center">

<img src="assets/sanjeevani%20logo.png" alt="Sanjeevani AI Logo" width="260"/>

# Sanjeevani AI

### Detecting risk before it becomes an emergency.

A hybrid classical-quantum machine learning system for early heart disease risk screening — built for **IEEE Region 10 QAI-Lead 2026**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Machine%20Learning-6929C4?style=flat-square&logo=qiskit&logoColor=white)](https://qiskit-community.github.io/qiskit-machine-learning)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Classical%20ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/IEEE%20R10-QAI--Lead%202026-0A66C2?style=flat-square)](#team)

Sanjeevani AI compares a classical AI model against **two independent quantum machine learning approaches** (a Quantum Kernel SVM and a Variational Quantum Classifier) on the same clinical dataset, with a live, interactive demo where all three models predict risk side by side in real time.

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Why "Sanjeevani"](#why-sanjeevani)
- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Technical Decisions](#key-technical-decisions)
- [Limitations & Honest Disclosures](#limitations--honest-disclosures)
- [Future Work](#future-work)
- [References](#references)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [License](#license)

---

## Overview

Heart disease is one of the leading causes of death worldwide, and early risk screening can meaningfully change outcomes. Sanjeevani AI explores two questions at once:

1. Can a lightweight, instant risk-screening tool built on standard clinical features flag at-risk patients accurately?
2. Does quantum machine learning — specifically quantum kernel methods and variational quantum classifiers — offer any advantage over classical ML on this kind of small, tabular clinical dataset?

Rather than answering (2) with a single quantum model and calling it done, this project builds **two distinct quantum approaches** and honestly compares both against classical baselines under matched conditions.

---

## Why "Sanjeevani"

*Sanjeevani* is the mythological life-restoring herb from the Ramayana, brought by Hanuman to revive Lakshman before it was too late. The name reflects the project's core idea: **catching risk early, before it becomes a crisis** — not curing after collapse, but screening before it.

---

## Problem Statement

> **Early disease-risk screening (heart disease).**
> AI: train a classifier on a small tabular dataset (UCI heart) and expose it as a live form that returns a risk score.
> Quantum: run a Variational Quantum Classifier or quantum-kernel SVM (Qiskit ML) on 2–4 selected features and compare accuracy against the classical model.
> Real-time demo: enter values → instant risk + the quantum-vs-classical comparison.

This project satisfies every part of this brief, and extends it by implementing **both** a Quantum Kernel SVM *and* a VQC rather than just one.

---

## Project Architecture

```
Patient Data (13 clinical features)
        │
        ├──► Classical SVM (13 features)  ──► Risk Score
        │
        ├──► Feature Selection (Random Forest importance)
        │           │
        │           ▼
        │    Top 4 Features (cp, thal, thalach, oldpeak)
        │           │
        │    ┌──────┴──────┐
        │    ▼             ▼
        │  Classical SVM   Quantum Encoding (ZZFeatureMap, 4 qubits)
        │  (4 features)         │
        │                  ┌────┴────┐
        │                  ▼         ▼
        │           Quantum Kernel   VQC
        │           SVM (QSVC)       (Variational)
        │                  │         │
        └──────────────────┴─────────┘
                       │
                       ▼
         Live Comparison Demo (Streamlit)
         All models predict the same patient, side by side
```

---

## Dataset

**UCI Cleveland Heart Disease Dataset**
- 297–303 patients (rows with missing values dropped)
- 13 clinical features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia result
- Binary target: disease present / absent
- Source: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

## Methodology

### 1. Data Preparation
- Cleaned raw UCI file (fixed missing headers, converted `?` to proper missing values, binarized target from 0–4 severity scale to 0/1)
- Dropped rows with missing values
- 80/20 stratified train-test split
- Standardized all features (`StandardScaler`)

### 2. Feature Selection
- Trained a Random Forest purely to rank feature importance
- Selected the **top 4 features** — quantum circuits at today's qubit counts can only realistically handle a small number of inputs, so this reduction is a deliberate, principled choice, not an arbitrary one

### 3. Classical AI Model (Deliverable 1)
- SVM (RBF kernel), trained twice:
  - Once on all 13 features (best-case classical baseline)
  - Once on the *same* 4 features the quantum models use (for a fair, apples-to-apples comparison)
- Hyperparameters (`C`, `gamma`) tuned via 5-fold cross-validated `GridSearchCV`, using only the training set

### 4. Quantum Modules (Deliverable 2)

**Quantum Kernel SVM (QSVC)**
- Features encoded into a 4-qubit `ZZFeatureMap` (2 repetitions, linear entanglement)
- Similarity between patients computed via quantum state overlap (fidelity), replacing classical Euclidean distance
- Trained via `FidelityQuantumKernel` + `QSVC` on Qiskit's `AerSimulator`
- `C` hyperparameter tuned via cross-validation (found default C=1 was already optimal)

**Variational Quantum Classifier (VQC)**
- Same `ZZFeatureMap` encoding for a fair comparison with QSVC
- Trainable `RealAmplitudes` ansatz, optimized via `COBYLA`
- Random seed fixed via `algorithm_globals.random_seed` for reproducibility

### 5. Diagnostics
Both quantum models were checked for overfitting by comparing train vs. test accuracy — not just reported blind:
- **QSVC** showed a persistent ~22-point train-test gap, even after cross-validated regularization tuning — indicating the gap stems from the feature map's expressiveness relative to the dataset size, not insufficient regularization
- **VQC** showed a train-test gap of under 2 points — a much more reliable generalization profile at this qubit scale

### 6. Comparison (Deliverable 4)
All four models — Classical SVM (13 feat), Classical SVM (4 feat), QSVC, VQC — are benchmarked on identical accuracy, precision, recall, and F1 metrics and visualized in a single comparison chart.

### 7. Live Demo (Deliverable 3)
A Streamlit application where a user enters patient values (or loads a real example patient with one click) and instantly sees all three models' risk predictions as color-coded gauge charts, side by side.

---

## Results

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Classical SVM | 13 (all) | **86.67%** | 91.67% | 78.57% | 84.62% |
| Classical SVM | 4 (same as quantum) | 81.67% | 86.96% | 71.43% | 78.43% |
| Quantum Kernel SVM (QSVC) | 4 | 60.00% | 58.33% | 50.00% | 53.85% |
| VQC | 4 | 68.33% | 64.52% | 71.43% | 67.80% |

**Key finding:** Classical SVM outperforms both quantum approaches on this dataset, even under matched feature conditions. Between the two quantum methods, **VQC generalizes considerably more reliably than QSVC** — a genuinely interesting comparative result in its own right, and one most single-quantum-model projects wouldn't surface at all.

This is consistent with current literature: a 2025 *npj Digital Medicine* systematic review of 169 studies found no consistent quantum advantage yet in digital health, and most published quantum ML healthcare research to date relies on simulators and small datasets, much like this project.

---

## Live Demo

The Streamlit app (`app/streamlit_demo_pro.py`) provides:
- A 13-field patient input form with dynamically-generated labels (always reflects the true category values in the dataset, not hardcoded assumptions)
- **One-click example patients** — load a real high-risk or low-risk case from the dataset instantly, for a fast, reliable live demo
- Three color-coded speedometer gauges (Classical / QSVC / VQC), green-yellow-red risk zones
- An agreement/disagreement indicator across all three models
- An expandable full comparison chart and written summary

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sanjeevani-ai.git
cd sanjeevani-ai

python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` should include:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
qiskit
qiskit-machine-learning
qiskit-algorithms
streamlit
plotly
joblib
```

---

## Usage

Run the full pipeline in order:

```bash
python 01_data_prep_and_viz.py    # clean data, select top 4 features, generate EDA charts
python 02_classical_model.py      # train + tune classical SVM baselines
python 03_quantum_model.py        # train Quantum Kernel SVM (QSVC)
python 04_vqc_model.py            # train Variational Quantum Classifier (VQC)
python 05_comparison.py           # generate final 4-model comparison chart

streamlit run app/streamlit_demo_pro.py   # launch the live demo
```

Each script saves its outputs (`models/`, `results/`) so later steps and the demo can load them without retraining.

---

## Project Structure

```
sanjeevani-ai/
├── app/
│   └── streamlit_demo_pro.py              # live interactive demo
├── assets/
│   └── sanjeevani logo.png                # project logo
├── data/
│   ├── heart.csv                          # cleaned UCI Cleveland dataset
│   └── prepared_data.npz                  # train/test splits, scaled features
├── models/
│   ├── classical_4feat_model.joblib
│   ├── classical_full_model.joblib
│   ├── quantum_angle_scaler.joblib
│   ├── quantum_qsvc_model.joblib
│   ├── vqc_angle_scaler.joblib
│   ├── vqc_model.joblib
│   └── vqc_model.model                    # saved via Qiskit's native serializer
├── results/
│   ├── 01_target_balance.png
│   ├── 02_age_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_importance.png
│   ├── 05_classical_metrics_comparison.png
│   ├── 06_classical_confusion_matrix.png
│   ├── 07_classical_confusion_matrix_13feat.png
│   ├── 08_quantum_feature_map_circuit.png
│   ├── 09_full_comparison_all_models.png
│   ├── 10_quantum_confusion_matrix.png
│   ├── 11_vqc_confusion_matrix.png
│   ├── 12_full_comparison_with_vqc.png
│   ├── classical_results.json
│   ├── comparison_chart.png
│   ├── comparison_summary.md
│   ├── quantum_results.json
│   └── vqc_results.json
├── .gitignore
├── 01_data_prep_and_viz.py                # clean data, select top 4 features, generate EDA charts
├── 02_classical_model.py                  # train + tune classical SVM baselines
├── 03_quantum_model.py                    # train Quantum Kernel SVM (QSVC)
├── 04_vqc_model.py                        # train Variational Quantum Classifier (VQC)
├── 05_comparison.py                       # generate final 4-model comparison chart
├── LICENSE
├── prepare_data.py                        # raw UCI file → clean CSV fix-up script
├── quantum_sandbox.py                     # scratch/experimentation notebook script for quantum circuits
├── README.md
└── requirements.txt
```

> 💡 **Note:** the logo file currently has a space in its name (`sanjeevani logo.png`). It works fine on GitHub as shown above, but if you'd rather avoid spaces in filenames (cleaner for CLI/scripting), rename it to `sanjeevani_logo.png` and update the two `<img src="...">` tags near the top and bottom of this README to match.

---

## Key Technical Decisions

- **Why only 4 features for quantum models?** Quantum circuits at today's simulator/hardware scale become impractically slow and noisy well before 13 qubits. Features were chosen via Random Forest importance ranking — a principled, reproducible selection, not an arbitrary one.
- **Why compare against a 4-feature classical model, not just the 13-feature one?** Comparing a 4-qubit quantum model against a 13-feature classical model isn't a fair fight — classical would look artificially better purely from having more information. The 4-feature classical baseline isolates the actual quantum-vs-classical question.
- **Why both QSVC and VQC?** These are two fundamentally different quantum ML paradigms — kernel-based vs. variational — and they behave differently in practice, as this project's own results show (VQC generalizes better than QSVC here). Building both surfaces a genuine comparative finding a single-quantum-model project would miss entirely.
- **Why simulator-only, no real quantum hardware?** Consistent with the hackathon's defined scope (small qubit counts, simulator only) and standard practice in current QML research literature.

---

## Limitations & Honest Disclosures

- Training set for quantum models was subsampled (120–180 patients) to keep runtime feasible on classical hardware simulating quantum circuits — quantum kernel computation scales quadratically with sample count.
- QSVC shows a persistent overfitting gap that cross-validated `C` tuning did not resolve — attributed to the feature map's expressiveness relative to dataset size, not a bug in the pipeline.
- All quantum computation runs on Qiskit's `AerSimulator`, not real quantum hardware.
- Results are based on a single train/test split; a full k-fold cross-validated evaluation would give tighter confidence intervals but was out of scope for the hackathon timeline.

---

## Future Work

- Test on real IBM Quantum hardware via Qiskit Runtime
- Explore alternative feature maps (e.g., `PauliFeatureMap`) and ansätze for VQC
- Expand to additional early-screening use cases (diabetes, stroke risk) using the same pipeline
- Full k-fold cross-validated evaluation for tighter result confidence

---

## References

1. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212.
2. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
3. Gupta, R.S., et al. (2025). A systematic review of quantum machine learning for digital health. *npj Digital Medicine*.
4. Qiskit Machine Learning Documentation. https://qiskit-community.github.io/qiskit-machine-learning

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Qiskit](https://img.shields.io/badge/-Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=for-the-badge)
![Joblib](https://img.shields.io/badge/-Joblib-000000?style=for-the-badge)

</div>

Python · Pandas · NumPy · Scikit-learn · Qiskit · Qiskit Machine Learning · Streamlit · Plotly · Matplotlib · Seaborn · Joblib

---

## Team

**[LogicLords]**
[Aditi Gaikwad,Ketki Landge]
[MIT ADT University]

Built for IEEE Region 10 Educational Activities — Quantum & AI Leadership Program for Women in Engineering (QAI-Lead 2026), in collaboration with IEEE Women in Engineering (WIE) and Student Branches.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center">

---

<img src="assets/sanjeevani%20logo.png" alt="Sanjeevani AI" width="80"/>

**Sanjeevani AI** · *Detecting risk before it becomes an emergency.*

</div>