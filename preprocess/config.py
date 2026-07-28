import os


MIMIC_PATH = "/media/ubuntu/Student/fxy/PatientStateLearning/data/mimic/mimic-iv-ed-2.2"


TRIAGE_PATH = os.path.join(
    MIMIC_PATH,
    "triage.csv"
)


EDSTAY_PATH = os.path.join(
    MIMIC_PATH,
    "edstays.csv"
)


PATIENT_PATH = os.path.join(
    MIMIC_PATH,
    "patients.csv"
)


VITAL_PATH = os.path.join(
    MIMIC_PATH,
    "vitalsign.csv"
)


DIAGNOSIS_PATH = os.path.join(
    MIMIC_PATH,
    "diagnosis.csv"
)

