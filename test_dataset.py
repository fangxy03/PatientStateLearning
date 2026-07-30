from dataset.clinical_dataset import ClinicalDataset


dataset = ClinicalDataset(
    "data/processed/patient_observation.csv"
)


print("Dataset size:")
print(len(dataset))


patient,stay_id = dataset[0]


print("================")

print("stay_id:")
print(stay_id)


print("================")

print("patient:")

for k,v in patient.items():

    print(k,":",v)