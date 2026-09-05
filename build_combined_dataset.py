"""
Downloads and combines all 4 UCI Heart Disease sites
(Cleveland, Hungarian, Switzerland, VA Long Beach) into one dataset.

Run this ONCE to upgrade from 297 rows to ~900+ rows.
"""

import pandas as pd
import numpy as np
import urllib.request
import os


# ============================================================
# COLUMN NAMES
# ============================================================
cols = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


# ============================================================
# UCI DATASET SOURCES
# ============================================================
sources = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}


# ============================================================
# CREATE RAW DATA DIRECTORY
# ============================================================
os.makedirs("data/raw_sites", exist_ok=True)


# ============================================================
# DOWNLOAD ALL 4 SITES
# ============================================================
all_dfs = []

for site, url in sources.items():

    local_path = f"data/raw_sites/{site}.data"

    print(f"Downloading {site} ...")

    urllib.request.urlretrieve(url, local_path)

    df = pd.read_csv(
        local_path,
        header=None,
        names=cols,
        na_values="?"
    )

    df["source_site"] = site

    print(f" -> {site}: {df.shape[0]} rows")

    all_dfs.append(df)


# ============================================================
# COMBINE ALL SITES
# ============================================================
combined = pd.concat(
    all_dfs,
    ignore_index=True
)

print(
    f"\nTotal rows before cleaning: "
    f"{combined.shape[0]}"
)


# ============================================================
# BINARIZE TARGET
# 0 = no disease
# 1-4 = disease present
# ============================================================
combined["target"] = (
    combined["target"] > 0
).astype(int)


# ============================================================
# KNOWN DATA QUALITY ISSUE
# chol=0 is impossible, treat as missing
# ============================================================
combined.loc[
    combined["chol"] == 0,
    "chol"
] = np.nan


# ============================================================
# REMOVE EXACT DUPLICATE PATIENT ROWS
# ACROSS SITES
# ============================================================
before = combined.shape[0]

combined = combined.drop_duplicates(
    subset=cols
)

print(
    f"Removed {before - combined.shape[0]} duplicate rows"
)


# ============================================================
# CHECK MISSING VALUES BEFORE CLEANING
# ============================================================
print(
    "\nMissing values per column BEFORE "
    "dropping + imputation:"
)

print(combined.isnull().sum())


# ============================================================
# DROP FEATURES THAT ARE TOO SPARSE ACROSS SITES
#
# ca, thal, slope are heavily missing outside Cleveland.
# Imputing most of these values would mean fabricating data.
# ============================================================
missing_pct = (
    combined.isnull().mean() * 100
)

print("\nMissing % per column:")
print(missing_pct.round(1))


# ============================================================
# DROP THRESHOLD
# Drop columns missing in MORE THAN 40% of rows.
# NEVER drop target.
# ============================================================
DROP_THRESHOLD = 40

cols_to_drop = (
    missing_pct[
        missing_pct > DROP_THRESHOLD
    ]
    .index
    .tolist()
)

cols_to_drop = [
    c for c in cols_to_drop
    if c != "target"
]


print(
    f"\nDropping columns with "
    f">{DROP_THRESHOLD}% missing: "
    f"{cols_to_drop}"
)


# ============================================================
# DROP SPARSE FEATURES
# ============================================================
combined = combined.drop(
    columns=cols_to_drop
)


# ============================================================
# IMPUTE ONLY THE REMAINING FEATURES
# ============================================================
remaining_numeric_cols = [
    c
    for c in combined.columns
    if c not in ("target", "source_site")
]


for c in remaining_numeric_cols:

    combined[c] = combined[c].fillna(
        combined[c].median()
    )


# ============================================================
# VERIFY MISSING VALUES AFTER CLEANING
# ============================================================
print(
    "\nMissing values per column AFTER "
    "dropping + imputation:"
)

print(combined.isnull().sum())


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================
print(
    f"\nFinal combined shape: "
    f"{combined.shape}"
)


print("\nRows per site:")
print(
    combined["source_site"].value_counts()
)


print("\nTarget distribution:")
print(
    combined["target"].value_counts()
)


# ============================================================
# REMOVE source_site BEFORE SAVING
# ============================================================
final = combined.drop(
    columns=["source_site"]
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================
final.to_csv(
    "data/heart.csv",
    index=False
)


print(
    "\nSaved combined dataset to "
    "data/heart.csv"
)


# ============================================================
# SHOW FINAL FEATURES
# ============================================================
print(
    f"\nFinal feature columns kept: "
    f"{[c for c in final.columns if c != 'target']}"
)


print(
    "\nOriginal Cleveland-only backup is at "
    "data/heart_cleveland_only_BACKUP.csv"
)