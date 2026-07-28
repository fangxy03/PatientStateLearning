import pandas as pd
from config import *


def check_table(path,name):

    print("\n===================")
    print(name)

    df=pd.read_csv(path)

    print(df.head())

    print("\nshape:")
    print(df.shape)

    print("\ncolumns:")
    print(df.columns)

    print("\nmissing:")
    print(df.isnull().mean().sort_values(
        ascending=False
    ).head(10))


    return df



triage = check_table(
    TRIAGE_PATH,
    "TRIAGE"
)


ed = check_table(
    EDSTAY_PATH,
    "EDSTAYS"
)


#patients = check_table(
#    PATIENT_PATH,
#    "PATIENTS"
#)


vital = check_table(
    VITAL_PATH,
    "VITALS"
)


