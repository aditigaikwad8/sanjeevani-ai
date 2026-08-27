import numpy as np, json, time, joblib
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 42

data = np.load("data/prepared_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = list(data["feature_names"])
top4_features = list(data["top4_features"])
top4_idx = [feature_names.index(f) for f in top4_features]

X_train_4 = X_train[:, top4_idx]
X_test_4 = X_test[:, top4_idx]

angle_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_train_q = angle_scaler.fit_transform(X_train_4)
X_test_q = angle_scaler.transform(X_test_4)

n_qubits = X_train_q.shape[1]

# Same encoding circuit as the Quantum Kernel SVM, for a fair comparison
feature_map = zz_feature_map(feature_dimension=n_qubits, reps=2, entanglement="linear")

# The trainable part: an ansatz whose rotation angles get optimized during training
ansatz = real_amplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")

# Keep training set small — VQC trains iteratively, so this keeps runtime reasonable
max_train_samples = 120
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_q), max_train_samples, replace=False)
X_train_q_sub = X_train_q[idx]
y_train_sub = y_train[idx]

optimizer = COBYLA(maxiter=100)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
)

t0 = time.time()
vqc.fit(X_train_q_sub, y_train_sub)
train_time = time.time() - t0
print(f"VQC training took {train_time:.1f} seconds")

t0 = time.time()
preds = vqc.predict(X_test_q)
infer_time = time.time() - t0
print(f"VQC inference took {infer_time:.1f} seconds")

metrics = {
    "model": f"VQC ({n_qubits} qubits)",
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision": round(precision_score(y_test, preds), 4),
    "recall": round(recall_score(y_test, preds), 4),
    "f1": round(f1_score(y_test, preds), 4),
}
print(metrics)

with open("results/vqc_results.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved VQC results to results/vqc_results.json")

joblib.dump(angle_scaler, "models/vqc_angle_scaler.joblib")
print("Saved VQC angle scaler.")

try:
    joblib.dump(vqc, "models/vqc_model.joblib")
    print("Saved VQC model.")
except Exception as e:
    print("Could not save VQC model object (known qiskit-machine-learning limitation):", e)
    print("This does not affect your results — metrics and charts are already saved.")

# ============================================================
# VISUALIZATION SECTION
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. Confusion matrix for VQC
cm_vqc = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_vqc, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix — VQC (4 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/11_vqc_confusion_matrix.png", dpi=150)
plt.close()

# 2. Full 4-way comparison: classical (13 & 4 feat), Quantum Kernel SVM, VQC
with open("results/classical_results.json") as f:
    classical_results = json.load(f)
with open("results/quantum_results.json") as f:
    qsvc_results = json.load(f)

all_results = classical_results + [qsvc_results, metrics]
results_df = pd.DataFrame(all_results).set_index("model")

results_df.plot(kind="bar", figsize=(11, 5), colormap="plasma")
plt.title("Classical vs Quantum Kernel SVM vs VQC: Full Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("results/12_full_comparison_with_vqc.png", dpi=150)
plt.close()

print("\nSaved 2 charts to results/ folder:")
print(" - 11_vqc_confusion_matrix.png")
print(" - 12_full_comparison_with_vqc.png")

train_preds = vqc.predict(X_train_q_sub)
print("Train accuracy:", accuracy_score(y_train_sub, train_preds))
print("Test accuracy:", accuracy_score(y_test, preds))