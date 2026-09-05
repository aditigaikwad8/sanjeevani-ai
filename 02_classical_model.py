import numpy as np
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# LOAD PREPARED DATA
# ============================================================

data = np.load("data/prepared_data.npz", allow_pickle=True)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

feature_names = list(data["feature_names"])
top4_features = list(data["top4_features"])

top4_idx = [feature_names.index(f) for f in top4_features]

print("============================================================")
print("CLASSICAL SVM MODEL TRAINING")
print("============================================================")

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"Total features:   {X_train.shape[1]}")
print(f"Top-4 features:   {top4_features}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, X_te, y_te, name):

    preds = model.predict(X_te)

    m = {
        "model": name,
        "accuracy": round(
            accuracy_score(y_te, preds), 4
        ),
        "precision": round(
            precision_score(y_te, preds, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_te, preds, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_te, preds, zero_division=0), 4
        )
    }

    print("\n------------------------------------------------------------")
    print(f"Evaluation: {name}")
    print("------------------------------------------------------------")

    print(f"Accuracy : {m['accuracy']}")
    print(f"Precision: {m['precision']}")
    print(f"Recall   : {m['recall']}")
    print(f"F1 Score : {m['f1']}")

    return m


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# SVM HYPERPARAMETER GRID
# ============================================================
#
# GridSearchCV will test different values of:
#
# C     -> controls regularization / overfitting
# gamma -> controls influence of individual training points
#
# cv=5 means 5-fold cross-validation.
#
# IMPORTANT:
# The TEST SET is NOT used during this tuning process.
#
# ============================================================

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
}


# ============================================================
# 1. OPTIMIZED SVM — 13 FEATURES
# ============================================================

print("\n============================================================")
print("TUNING SVM USING ALL 13 FEATURES")
print("============================================================")

grid_full = GridSearchCV(
    estimator=SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# IMPORTANT:
# Only X_train and y_train are used here.
grid_full.fit(X_train, y_train)

print("\nBest parameters for 13-feature SVM:")
print(grid_full.best_params_)

print(
    "Best 5-fold cross-validation accuracy:",
    round(grid_full.best_score_, 4)
)

# Best SVM selected by cross-validation
svc_full = grid_full.best_estimator_


# ============================================================
# FINAL TEST EVALUATION — 13 FEATURES
# ============================================================

results.append(
    evaluate(
        svc_full,
        X_test,
        y_test,
        "SVM (13 features)"
    )
)


# ============================================================
# 2. PREPARE THE SAME 4 FEATURES USED BY QUANTUM MODEL
# ============================================================

X_train_4 = X_train[:, top4_idx]
X_test_4 = X_test[:, top4_idx]

print("\n============================================================")
print("4-FEATURE DATA")
print("============================================================")

print("Selected features:")
for feature in top4_features:
    print(" -", feature)

print(f"\nTraining shape: {X_train_4.shape}")
print(f"Testing shape : {X_test_4.shape}")


# ============================================================
# 3. OPTIMIZED SVM — SAME 4 FEATURES AS QUANTUM MODEL
# ============================================================

print("\n============================================================")
print("TUNING SVM USING SAME 4 FEATURES AS QUANTUM MODEL")
print("============================================================")

grid_4 = GridSearchCV(
    estimator=SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# IMPORTANT:
# Again, only training data is used for tuning.
grid_4.fit(X_train_4, y_train)

print("\nBest parameters for 4-feature SVM:")
print(grid_4.best_params_)

print(
    "Best 5-fold cross-validation accuracy:",
    round(grid_4.best_score_, 4)
)

# Best SVM selected by cross-validation
svc_4 = grid_4.best_estimator_


# ============================================================
# FINAL TEST EVALUATION — 4 FEATURES
# ============================================================

results.append(
    evaluate(
        svc_4,
        X_test_4,
        y_test,
        "SVM (same 4 features as quantum)"
    )
)


# ============================================================
# SAVE TRAINED MODELS
# ============================================================

joblib.dump(
    svc_full,
    "models/classical_full_model.joblib"
)

joblib.dump(
    svc_4,
    "models/classical_4feat_model.joblib"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/classical_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n============================================================")
print("FINAL CLASSICAL SVM RESULTS")
print("============================================================")

for result in results:

    print(
        f"\n{result['model']}"
    )

    print(
        f"Accuracy : {result['accuracy']}"
    )

    print(
        f"Precision: {result['precision']}"
    )

    print(
        f"Recall   : {result['recall']}"
    )

    print(
        f"F1 Score : {result['f1']}"
    )


# ============================================================
# VISUALIZATION SECTION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


# ============================================================
# 1. BAR CHART
# ============================================================

results_df = pd.DataFrame(results).set_index("model")

results_df.plot(
    kind="bar",
    figsize=(8, 5),
    colormap="viridis"
)

plt.title(
    "Classical SVM Performance: 13 Features vs Top-4 Features"
)

plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(
    rotation=15
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

plt.savefig(
    "results/05_classical_metrics_comparison.png",
    dpi=150
)

plt.close()


# ============================================================
# 2. CONFUSION MATRIX — 4 FEATURES
# ============================================================

preds_4 = svc_4.predict(X_test_4)

cm = confusion_matrix(
    y_test,
    preds_4
)

plt.figure(
    figsize=(5, 4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(
    "Confusion Matrix — Classical SVM (4 Features)"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "results/06_classical_confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 3. CONFUSION MATRIX — 13 FEATURES
# ============================================================

preds_full = svc_full.predict(X_test)

cm_full = confusion_matrix(
    y_test,
    preds_full
)

plt.figure(
    figsize=(5, 4)
)

sns.heatmap(
    cm_full,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(
    "Confusion Matrix — Classical SVM (13 Features)"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "results/07_classical_confusion_matrix_13feat.png",
    dpi=150
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================================")
print("FILES SAVED")
print("============================================================")

print("\nModels:")
print(" - models/classical_full_model.joblib")
print(" - models/classical_4feat_model.joblib")

print("\nResults:")
print(" - results/classical_results.json")

print("\nCharts:")
print(" - results/05_classical_metrics_comparison.png")
print(" - results/06_classical_confusion_matrix.png")
print(" - results/07_classical_confusion_matrix_13feat.png")

print("\n============================================================")
print("CLASSICAL SVM TRAINING COMPLETED SUCCESSFULLY")
print("============================================================")
