import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
df = df.drop(columns=["CustomerID"])

# correct Gender(Fe Male,Female) /Martial Status (Single/unmarried)

# Drop rows with missing values
df = df.dropna()

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Train-test split (stratified)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits to current directory (GitHub Actions uploads these as artifacts)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete.")
print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
print(f"Class balance (train):\n{ytrain.value_counts(normalize=True)}")
