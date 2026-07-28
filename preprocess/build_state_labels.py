import pandas as pd
import os


input_path = "data/processed/patient_observation.csv"

output_path = "data/processed/state_labels.csv"


print("Loading data...")

df = pd.read_csv(input_path)


print(df.shape)



# ==========================
# 1. Circulation State
# Hemodynamic instability
# ==========================


circulation_condition1 = (
    df["sbp_min"] < 90
)


circulation_condition2 = (
    (df["heartrate_max"] > 120)
    &
    (df["sbp_mean"] < 100)
)


df["circulation_label"] = (
    circulation_condition1
    |
    circulation_condition2
).astype(int)



# ==========================
# 2. Infection State
# Infection / Sepsis risk
# ==========================


infection_temperature = (
    df["temperature_max"] > 38
)


infection_hr = (
    df["heartrate_max"] > 100
)


infection_rr = (
    df["resprate_max"] > 20
)


infection_text = (
    df["chiefcomplaint"]
    .fillna("")
    .str.lower()
    .str.contains(
        "fever|infection|sepsis|chill|pneumonia",
        regex=True
    )
)


infection_score = (
    infection_temperature.astype(int)
    +
    infection_hr.astype(int)
    +
    infection_rr.astype(int)
    +
    infection_text.astype(int)
)


df["infection_label"] = (
    infection_score >= 2
).astype(int)



# ==========================
# 3. Organ Dysfunction State
# ==========================


organ_condition1 = (
    df["o2sat_min"] < 92
)


organ_condition2 = (
    (df["o2sat_min"] < 94)
    &
    (df["resprate_max"] > 30)
)


df["organ_label"] = (
    organ_condition1
    |
    organ_condition2
).astype(int)