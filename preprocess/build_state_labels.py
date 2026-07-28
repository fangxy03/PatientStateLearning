import pandas as pd
import os


input_path = "../data/processed/patient_observation.csv"

output_path = "../data/processed/state_labels.csv"


print("Loading data...")

df = pd.read_csv(input_path)


print(df.shape)



# ==========================
# 1. Circulation State
# ==========================

# 低血压作为循环异常信号

df["circulation_label"] = (
    (df["sbp_min"] < 90)
).astype(int)



# ==========================
# 2. Infection State
# ==========================

# 发热作为感染风险信号

df["infection_label"] = (
    (df["temperature_max"] > 38)
).astype(int)



# ==========================
# 3. Organ Dysfunction State
# ==========================

# 低氧或者呼吸异常

df["organ_label"] = (
    (df["o2sat_min"] < 92)
    |
    (df["resprate_max"] > 30)
).astype(int)



# 保存

labels = df[
    [
        "stay_id",
        "circulation_label",
        "infection_label",
        "organ_label"
    ]
]


print(labels.head())

print("================")
print(labels["circulation_label"].value_counts())
print(labels["infection_label"].value_counts())
print(labels["organ_label"].value_counts())


labels.to_csv(
    output_path,
    index=False
)


print("Finished!")