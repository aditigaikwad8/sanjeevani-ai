import numpy as np, json, time, joblib
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

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
feature_map = zz_feature_map(feature_dimension=n_qubits, reps=2, entanglement="linear")
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

max_train_samples = 180
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_q), max_train_samples, replace=False)
X_train_q_sub = X_train_q[idx]
y_train_sub = y_train[idx]

# ============================================================
# HYPERPARAMETER TUNING — fixes the overfitting seen earlier
# (train accuracy 82% vs test accuracy 60% with default C=1.0).
# Lower C = stronger regularization = less overfitting.
# cv=3, not 5 — each fold recomputes the full quantum kernel,
# so 3 keeps runtime manageable. Only X_train_q_sub/y_train_sub
# are used here — the test set is never touched during tuning.
# NOTE: this step is expensive — 5 C-values x 3 folds = 15 full
# quantum kernel fits. Expect ~10-15 minutes. Let it run.
# ============================================================

print("============================================================")
print("TUNING QUANTUM KERNEL SVM (this will take ~10-15 minutes)")
print("============================================================")

param_grid = {"C": [0.1, 0.5, 1, 2, 5]}

grid = GridSearchCV(
    estimator=QSVC(quantum_kernel=quantum_kernel),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2
)

t0 = time.time()
grid.fit(X_train_q_sub, y_train_sub)
tune_time = time.time() - t0
print(f"\nHyperparameter tuning took {tune_time:.1f} seconds")

print("Best C:", grid.best_params_["C"])
print("Best cross-validation accuracy:", round(grid.best_score_, 4))

qsvc = grid.best_estimator_

# ============================================================
# FINAL TRAIN — refit the best model on the full subsample
# (GridSearchCV already does this internally via refit=True by
# default, but we time it separately here for a clean, reportable
# "training time" number, matching the style of your other scripts)
# ============================================================

t0 = time.time()
qsvc.fit(X_train_q_sub, y_train_sub)
train_time = time.time() - t0
print(f"\nQuantum training took {train_time:.1f} seconds")

t0 = time.time()
preds = qsvc.predict(X_test_q)
infer_time = time.time() - t0
print(f"Quantum inference took {infer_time:.1f} seconds")

metrics = {
    "model": f"Quantum Kernel SVM ({n_qubits} qubits)",
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision": round(precision_score(y_test, preds, zero_division=0), 4),
    "recall": round(recall_score(y_test, preds, zero_division=0), 4),
    "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    "best_C": grid.best_params_["C"],
}
print(metrics)

with open("results/quantum_results.json", "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(qsvc, "models/quantum_qsvc_model.joblib")
joblib.dump(angle_scaler, "models/quantum_angle_scaler.joblib")
print("Saved quantum model + scaler.")

# ============================================================
# VISUALIZATION SECTION
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. Confusion matrix for the quantum model
cm_q = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_q, annot=True, fmt="d", cmap="Purples",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix — Quantum Kernel SVM (4 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/10_quantum_confusion_matrix.png", dpi=150)
plt.close()

# 2. Full 3-way comparison: 13-feature classical, 4-feature classical, 4-feature quantum
with open("results/classical_results.json") as f:
    classical_results = json.load(f)
all_results = classical_results + [metrics]
results_df = pd.DataFrame(all_results).set_index("model")
# Drop the extra best_C column before plotting — it's not a score metric
plot_df = results_df.drop(columns=["best_C"], errors="ignore")

plot_df.plot(kind="bar", figsize=(10, 5), colormap="plasma")
plt.title("Classical vs Quantum: Full Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("results/09_full_comparison_all_models.png", dpi=150)
plt.close()

print("\nSaved 2 charts to results/ folder:")
print(" - 09_full_comparison_all_models.png")
print(" - 10_quantum_confusion_matrix.png")

# ============================================================
# DIAGNOSTIC — check if the overfitting gap has closed
# ============================================================
train_preds = qsvc.predict(X_train_q_sub)
train_acc = accuracy_score(y_train_sub, train_preds)
test_acc = accuracy_score(y_test, preds)
print("\n============================================================")
print("OVERFITTING CHECK")
print("============================================================")
print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
print("Train - Test gap:", round(train_acc - test_acc, 4))
print("(Earlier run with default C=1.0: train 0.822, test 0.60, gap 0.222)")