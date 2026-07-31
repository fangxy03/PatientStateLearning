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

        if not csv_path.exists():
            fallback_path = repo_root / "data/processed/patient_observation.csv"
            if fallback_path.exists():
                csv_path = fallback_path
            else:
                raise FileNotFoundError(f"Dataset file not found: {csv_file}")

        self.data = pd.read_csv(csv_path)

        # 根据 CSV 实际路径向上寻找 mimic patients 目录
        csv_dir = csv_path.resolve().parent
        patients_candidates = [
            csv_dir / "../mimic/mimic-iv-ed-2.2/patients.csv",
            csv_dir / "../../data/mimic/mimic-iv-ed-2.2/patients.csv",
            repo_root / "data/mimic/mimic-iv-ed-2.2/patients.csv",
        ]
        patients_path = None
        for p in patients_candidates:
            if p.exists():
                patients_path = p
                break

        if patients_path is not None:
            patients = pd.read_csv(patients_path)
            patients = patients[["subject_id", "anchor_age"]]
            self.data = self.data.merge(patients, on="subject_id", how="left")
        else:
            if "anchor_age" not in self.data.columns:
                self.data["anchor_age"] = pd.NA

        print("Original rows:", len(self.data))

        self.data = self._clean_data()
        print("Cleaned rows:", len(self.data))

        # anchor_age 缺失补 0
        if "anchor_age" not in self.data.columns:
            self.data["anchor_age"] = 0
        else:
            self.data["anchor_age"] = self.data["anchor_age"].fillna(0).infer_objects(copy=False)

    def _clean_data(self):
        df = self.data.copy()

        # 确保关键列存在，缺失列则补 NA（不删除）
        expected_columns = [
            "temperature", "heartrate", "resprate", "o2sat",
            "sbp", "dbp", "chiefcomplaint", "pain", "acuity",
        ]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA

        # 仅剔除 acuity 缺失的样本，其他字段缺失一律保留
        before = len(df)
        df = df.dropna(subset=["acuity"]).copy()
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with missing acuity label")

        if "anchor_age" not in df.columns:
            df["anchor_age"] = pd.NA

        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        patient = {
            "age": int(row["anchor_age"]) if pd.notna(row["anchor_age"]) else 0,
            "temperature": float(row["temperature"]) if pd.notna(row["temperature"]) else 0.0,
            "heartrate": float(row["heartrate"]) if pd.notna(row["heartrate"]) else 0.0,
            "resprate": float(row["resprate"]) if pd.notna(row["resprate"]) else 0.0,
            "o2sat": float(row["o2sat"]) if pd.notna(row["o2sat"]) else 0.0,
            "sbp": float(row["sbp"]) if pd.notna(row["sbp"]) else 0.0,
            "dbp": float(row["dbp"]) if pd.notna(row["dbp"]) else 0.0,
            "chiefcomplaint": str(row["chiefcomplaint"]),
            "pain": self.convert_pain(row["pain"]),
        }

        acuity = row["acuity"]
        if pd.isna(acuity):
            raise KeyError("Missing acuity")

        label = torch.tensor(int(acuity) - 1, dtype=torch.long)
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
