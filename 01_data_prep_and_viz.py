import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
 
RANDOM_STATE = 42
 
df = pd.read_csv("data/heart.csv")
X = df.drop(columns=["target"])
y = df["target"]
feature_names = X.columns.tolist()
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)
 
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Feature importance ranking:\n", importances)
 
top4_features = importances.index[:4].tolist()
print("\nTop 4 features for quantum model:", top4_features)
 
np.savez("data/prepared_data.npz",
    X_train=X_train_scaled, X_test=X_test_scaled,
    y_train=y_train.values, y_test=y_test.values,
    feature_names=np.array(feature_names),
    top4_features=np.array(top4_features))
 
print("\nSaved to data/prepared_data.npz")
 
# ============================================================
# VISUALIZATION SECTION
# Saves a few simple charts into the results/ folder
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
os.makedirs("results", exist_ok=True)
 
# 1. Target class balance (how many sick vs healthy)
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df, palette=["#4C72B0", "#C44E52"])
plt.title("Heart Disease Cases (0 = No, 1 = Yes)")
plt.xlabel("Target")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("results/01_target_balance.png", dpi=150)
plt.close()
 
# 2. Age distribution, split by disease status
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True, palette=["#4C72B0", "#C44E52"])
plt.title("Age Distribution by Heart Disease Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/02_age_distribution.png", dpi=150)
plt.close()
 
# 3. Correlation heatmap (how features relate to each other and to target)
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/03_correlation_heatmap.png", dpi=150)
plt.close()
 
# 4. Feature importance bar chart (the ranking we printed earlier)
plt.figure(figsize=(7, 5))
importances.sort_values().plot(kind="barh", color="#55A868")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("results/04_feature_importance.png", dpi=150)
plt.close()
 
print("\nSaved 4 charts to results/ folder:")
print(" - 01_target_balance.png")
print(" - 02_age_distribution.png")
print(" - 03_correlation_heatmap.png")
print(" - 04_feature_importance.png")
 