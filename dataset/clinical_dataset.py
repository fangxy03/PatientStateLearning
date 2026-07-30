import torch
from torch.utils.data import Dataset
import pandas as pd


class ClinicalDataset(Dataset):

    def __init__(
        self,
        csv_file
    ):

        self.data = pd.read_csv(csv_file)


    def __len__(self):

        return len(self.data)



    def __getitem__(self,index):

        row=self.data.iloc[index]


        # patient basic information

        patient={

            "age":row["age"],

            "heartrate":row["heartrate"],

            "sbp":row["sbp"],

            "o2sat":row["o2sat"],

            "temperature":row["temperature"]

        }


        # ESI label

        label=torch.tensor(
            row["esi"],
            dtype=torch.long
        )


        return patient,label