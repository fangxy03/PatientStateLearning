from dataset.clinical_dataset import ClinicalDataset



dataset=ClinicalDataset(

"data/subset/patient_subset.csv"

)



print("Dataset size:")

print(len(dataset))



patient,label=dataset[0]


print("================")

print(patient)


print("================")

print("ESI label:")

print(label)