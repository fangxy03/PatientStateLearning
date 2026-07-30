import torch
from torch.utils.data import Dataset
import pandas as pd


class ClinicalDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]


        # ==========================
        # Patient information
        # ==========================

        patient = {

            "temperature":
                float(row["temperature"]),

            "heartrate":
                float(row["heartrate"]),

            "resprate":
                float(row["resprate"]),

            "o2sat":
                float(row["o2sat"]),

            "sbp":
                float(row["sbp"]),

            "dbp":
                float(row["dbp"]),

            "chiefcomplaint":
                str(row["chiefcomplaint"]),

            "pain":
                float(row["pain"]),

            "acuity":
                int(row["acuity"])

        }


        # 先暂时返回stay_id
        # 后面和ESI label合并

        stay_id = row["stay_id"]


        return patient, stay_id