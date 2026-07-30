from dataset.clinical_dataset import ClinicalDataset


dataset=ClinicalDataset(
    "data/processed/patient_observation.csv"
)


print(len(dataset))


patient,label=dataset[0]


print(patient)

print(label)