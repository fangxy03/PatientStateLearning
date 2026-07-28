import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PatientDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def load_patient_data(
        csv_path
):

    df = pd.read_csv(csv_path)


    # remove non-medical information

    drop_columns = [

        "subject_id",
        "stay_id",
        "hadm_id",

        "intime",
        "outtime",

        "chiefcomplaint",

        "disposition"

    ]


    df = df.drop(
        columns=[
            c for c in drop_columns
            if c in df.columns
        ]
    )


    # remove samples without label

    df = df.dropna(
        subset=["acuity"]
    )


    # label

    y = df["acuity"].astype(int)-1


    X = df.drop(
        columns=["acuity"]
    )


    # categorical encoding

    X = pd.get_dummies(
        X
    )


    # missing values

    X = X.fillna(
        X.median()
    )


    scaler = StandardScaler()

    X = scaler.fit_transform(
        X
    )


    return X, y.values
