import pandas as pd


obs = pd.read_csv(
    "data/processed/patient_observation.csv"
)


patients = pd.read_csv(
    "data/raw/patients.csv"
)


patients = patients[
    [
        "subject_id",
        "anchor_age"
    ]
]


df = obs.merge(
    patients,
    on="subject_id",
    how="left"
)


df.rename(
    columns={
        "anchor_age":"age"
    },
    inplace=True
)


df.to_csv(
    "data/processed/patient_observation_age.csv",
    index=False
)


print(df.head())

print(
    df["age"].isna().sum()
)