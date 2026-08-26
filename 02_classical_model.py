import numpy as np, json, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = np.load("data/prepared_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = list(data["feature_names"])
top4_features = list(data["top4_features"])
top4_idx = [feature_names.index(f) for f in top4_features]

def evaluate(model, X_te, y_te, name):
    preds = model.predict(X_te)
    m = {"model": name,
         "accuracy": round(accuracy_score(y_te, preds), 4),
         "precision": round(precision_score(y_te, preds), 4),
         "recall": round(recall_score(y_te, preds), 4),
         "f1": round(f1_score(y_te, preds), 4)}
    print(m)
    return m

results = []

svc_full = SVC(kernel="rbf", probability=True, random_state=42)
svc_full.fit(X_train, y_train)
results.append(evaluate(svc_full, X_test, y_test, "SVM (13 features)"))

X_train_4, X_test_4 = X_train[:, top4_idx], X_test[:, top4_idx]
svc_4 = SVC(kernel="rbf", probability=True, random_state=42)
svc_4.fit(X_train_4, y_train)
results.append(evaluate(svc_4, X_test_4, y_test, "SVM (same 4 features as quantum)"))

joblib.dump(svc_full, "models/classical_full_model.joblib")
joblib.dump(svc_4, "models/classical_4feat_model.joblib")
with open("results/classical_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# VISUALIZATION SECTION
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# 1. Bar chart comparing accuracy/precision/recall/F1 across both models
results_df = pd.DataFrame(results).set_index("model")
results_df.plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("Classical SVM Performance: 13 Features vs Top-4 Features")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("results/05_classical_metrics_comparison.png", dpi=150)
plt.close()

# 2. Confusion matrix for the 4-feature model (the one we'll compare to quantum)
preds_4 = svc_4.predict(X_test_4)
cm = confusion_matrix(y_test, preds_4)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix — Classical SVM (4 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/06_classical_confusion_matrix.png", dpi=150)
plt.close()

# 3. Confusion matrix for the 13-feature model (for comparison)
preds_full = svc_full.predict(X_test)
cm_full = confusion_matrix(y_test, preds_full)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_full, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix — Classical SVM (13 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/07_classical_confusion_matrix_13feat.png", dpi=150)
plt.close()

print("\nSaved 3 charts to results/ folder:")
print(" - 05_classical_metrics_comparison.png")
print(" - 06_classical_confusion_matrix.png (4-feature model)")
print(" - 07_classical_confusion_matrix_13feat.png (13-feature model)")

print("Saved models and results.")