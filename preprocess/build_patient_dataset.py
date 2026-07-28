import pandas as pd
from config import *


print("Loading triage...")
triage = pd.read_csv(TRIAGE_PATH)


print("Loading edstays...")
ed = pd.read_csv(EDSTAY_PATH)


print("Loading vitals...")
vitals = pd.read_csv(VITAL_PATH)



############################
# 1. vital aggregation
############################


vital_features = vitals.groupby(
    "stay_id"
).agg({

    "temperature":[
        "mean",
        "max",
        "min"
    ],

    "heartrate":[
        "mean",
        "max",
        "min"
    ],

    "resprate":[
        "mean",
        "max",
        "min"
    ],

    "o2sat":[
        "mean",
        "min"
    ],

    "sbp":[
        "mean",
        "min"
    ],

    "dbp":[
        "mean",
        "min"
    ]

})


# flatten columns

vital_features.columns=[
    "_".join(col)
    for col in vital_features.columns
]


vital_features.reset_index(
    inplace=True
)


print(vital_features.head())


############################
# 2. merge
############################


data = triage.merge(
    ed,
    on="stay_id",
    how="left"
)


data = data.merge(
    vital_features,
    on="stay_id",
    how="left"
)

# ==========================
# clean duplicated columns
# ==========================

if "subject_id_x" in data.columns:
    data = data.rename(
        columns={
            "subject_id_x": "subject_id"
        }
    )


if "subject_id_y" in data.columns:
    data = data.drop(
        columns=["subject_id_y"]
    )

print(data.shape)

print(data.head())



############################
# save
############################
# clean duplicated columns

data = data.rename(
    columns={
        "subject_id_x":"subject_id"
    }
)


if "subject_id_y" in data.columns:
    data=data.drop(
        columns=["subject_id_y"]
    )

data.to_csv(
    "data/processed/patient_observation.csv",
    index=False
)


print("Finished!")

