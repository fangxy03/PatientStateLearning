import pandas as pd

from config import *


# load

triage=pd.read_csv(
    TRIAGE_PATH
)


patients=pd.read_csv(
    PATIENT_PATH
)


ed=pd.read_csv(
    EDSTAY_PATH
)


# 合并急诊信息

data = triage.merge(
    ed,
    on="stay_id",
    how="left"
)


# 加入人口信息

data = data.merge(
    patients,
    on="subject_id",
    how="left"
)



print(data.shape)

print(data.head())


data.to_csv(
    "../data/processed/patient_dataset.csv",
    index=False
)

