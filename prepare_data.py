import pandas as pd

# The 14 column names, in the correct order
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]

# Read the RAW file (change the filename below to match what's in your data folder)
df = pd.read_csv("data/heart.csv", header=None, names=cols, na_values="?")

# Turn target into 0 (no disease) or 1 (disease), instead of 0-4
df["target"] = (df["target"] > 0).astype(int)

# Save the clean, proper version — this OVERWRITES your broken heart.csv
df.to_csv("data/heart.csv", index=False)

print("Saved data/heart.csv")
print(df.shape)
print(df["target"].value_counts())