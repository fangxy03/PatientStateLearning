import pandas as pd
import os


input_file = "data/processed/patient_observation.csv"


df = pd.read_csv(input_file)


print("Original size:")
print(len(df))


# 抽1000条测试
subset=df.sample(
    n=1000,
    random_state=42
)


os.makedirs(
    "data/subset",
    exist_ok=True
)


subset.to_csv(
    "data/subset/patient_subset.csv",
    index=False
)


print("Subset size:")
print(len(subset))


print("\nESI distribution:")
print(
    subset["acuity"].value_counts()
)