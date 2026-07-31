import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class ClinicalDataset(Dataset):

    def __init__(self, csv_file):
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path

        self.data = pd.read_csv(csv_path)

        patients_path = repo_root / "data/mimic/mimic-iv-ed-2.2/patients.csv"
        patients = pd.read_csv(patients_path)

        patients = patients[["subject_id", "anchor_age"]]
        patients = patients.rename(columns={"anchor_age": "age"})

        self.data = self.data.merge(patients, on="subject_id", how="left")

        print("Missing age:", self.data["age"].isna().sum())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        patient = {
            "age": int(row["age"]) if pd.notna(row["age"]) else 0,
            "temperature": float(row["temperature"]) if pd.notna(row["temperature"]) else 0.0,
            "heartrate": float(row["heartrate"]) if pd.notna(row["heartrate"]) else 0.0,
            "resprate": float(row["resprate"]) if pd.notna(row["resprate"]) else 0.0,
            "o2sat": float(row["o2sat"]) if pd.notna(row["o2sat"]) else 0.0,
            "sbp": float(row["sbp"]) if pd.notna(row["sbp"]) else 0.0,
            "dbp": float(row["dbp"]) if pd.notna(row["dbp"]) else 0.0,
            "chiefcomplaint": str(row["chiefcomplaint"]),
            "pain": self.convert_pain(row["pain"]),
        }

        label = torch.tensor(int(row["acuity"]) - 1, dtype=torch.long)
        return patient, label

    def convert_pain(self, value):
        if pd.isna(value):
            return 0.0

        try:
            return float(value)
        except Exception:
            pass

        mapping = {
            "none": 0,
            "mild": 3,
            "moderate": 5,
            "severe": 8,
            "unable": 10,
        }

        value = str(value).strip().lower()
        return mapping.get(value, 0.0)
