import torch
from torch.utils.data import Dataset
import pandas as pd



class ClinicalDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)



    def __len__(self):

        return len(self.data)



    def __getitem__(self,index):

        row=self.data.iloc[index]


        patient={


            # 基本信息

            "stay_id":
                int(row["stay_id"]),



            # 生命体征

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



            # 主诉

            "chiefcomplaint":
                str(row["chiefcomplaint"]),



            # 疼痛

            "pain":
                float(row["pain"]),



            # 急诊等级
            # 作为label，不给agent看

        }



        label=torch.tensor(

            int(row["acuity"])-1,

            dtype=torch.long

        )


        return patient,label