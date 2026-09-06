import numpy as np
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit_machine_learning.algorithms import VQC

print("="*60)
print("ENSEMBLE MODEL: Combining Classical + QSVC + VQC")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================
data = np.load("data/prepared_data.npz", allow_pickle=True)
X_test = data["X_test"]
y_test = data["y_test"]
feature_names = list(data["feature_names"])
top4_features = list(data["top4_features"])
top4_idx = [feature_names.index(f) for f in top4_features]
X_test_4 = X_test[:, top4_idx]

# ============================================================
# LOAD ALL THREE TRAINED MODELS
# ============================================================
classical_4feat = joblib.load("models/classical_4feat_model.joblib")
qsvc_model = joblib.load("models/quantum_qsvc_model.joblib")
quantum_scaler = joblib.load("models/quantum_angle_scaler.joblib")
vqc_model = VQC.from_dill("models/vqc_model.model")
vqc_scaler = joblib.load("models/vqc_angle_scaler.joblib")

quantum_input = quantum_scaler.transform(X_test_4)
vqc_input = vqc_scaler.transform(X_test_4)

# ============================================================
# HELPER: get a probability score from any model, with fallback
# ============================================================
def get_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            scores = model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        except Exception:
            return model.predict(X).astype(float)

classical_proba = get_proba(classical_4feat, X_test_4)
qsvc_proba = get_proba(qsvc_model, quantum_input)
vqc_proba = get_proba(vqc_model, vqc_input)

# ============================================================
# BUILD THREE ENSEMBLE VARIANTS
# ============================================================
quantum_only_proba = (qsvc_proba + vqc_proba) / 2
hybrid_equal_proba = (classical_proba + qsvc_proba + vqc_proba) / 3
# Weighted: classical gets more say since it's individually more accurate
hybrid_weighted_proba = (0.5 * classical_proba) + (0.25 * qsvc_proba) + (0.25 * vqc_proba)

def evaluate(proba, y_true, name):
    preds = (proba > 0.5).astype(int)
    m = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
    }
    print(f"\n{name}")
    print(f"  Accuracy : {m['accuracy']}")
    print(f"  Precision: {m['precision']}")
    print(f"  Recall   : {m['recall']}")
    print(f"  F1 Score : {m['f1']}")
    return m

results = []
results.append(evaluate(quantum_only_proba, y_test, "Ensemble: QSVC + VQC (quantum-only)"))
results.append(evaluate(hybrid_equal_proba, y_test, "Ensemble: Classical + QSVC + VQC (equal weight)"))
results.append(evaluate(hybrid_weighted_proba, y_test, "Ensemble: Classical + QSVC + VQC (weighted 50/25/25)"))

with open("results/ensemble_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/ensemble_results.json")

# ============================================================
# FULL COMPARISON AGAINST EVERYTHING ELSE
# ============================================================
with open("results/classical_results.json") as f:
    classical = json.load(f)
with open("results/quantum_results.json") as f:
    quantum = json.load(f)
with open("results/vqc_results.json") as f:
    vqc = json.load(f)

all_results = classical + [quantum, vqc] + results

print("\n" + "="*60)
print("FULL COMPARISON TABLE — ALL MODELS")
print("="*60)
for r in sorted(all_results, key=lambda x: -x["accuracy"]):
    print(f"{r['model']:<55} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

# ============================================================
# CHART
# ============================================================
import matplotlib.pyplot as plt
names = [r["model"] for r in all_results]
accs = [r["accuracy"] for r in all_results]
f1s = [r["f1"] for r in all_results]
x = np.arange(len(all_results))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, accs, width, label="Accuracy", color="#1f6fb2")
ax.bar(x + width/2, f1s, width, label="F1 Score", color="#f5a623")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("All Models Including Ensembles")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/13_ensemble_comparison.png", dpi=150)
print("\nSaved results/13_ensemble_comparison.png")