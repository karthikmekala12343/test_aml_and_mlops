import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

df = pd.read_csv(DATA_PATH)

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Dataset registered. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("Target distribution:")
print(df['ProdTaken'].value_counts())
