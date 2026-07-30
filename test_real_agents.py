from dataset.clinical_dataset import ClinicalDataset


from llm_agents.qwen_model import QwenModel

from llm_agents.circulation_agent import CirculationAgent
from llm_agents.infection_agent import InfectionAgent
from llm_agents.organ_agent import OrganAgent



print("================")
print("Loading Dataset")
print("================")


dataset = ClinicalDataset(
    "data/subset/patient_subset.csv"
)



print("Dataset size:",
      len(dataset))


# 取一个患者

patient,label = dataset[0]


print("\nPatient:")
print(patient)


print("\nESI label:")
print(label)



print("================")
print("Loading LLM")
print("================")


llm = QwenModel()



circ = CirculationAgent(llm)

inf = InfectionAgent(llm)

organ = OrganAgent(llm)



print("\n===== Circulation =====")


circ_output = circ.analyze(patient)

print(circ_output)



print("\n===== Infection =====")


inf_output = inf.analyze(patient)

print(inf_output)



print("\n===== Organ =====")


organ_output = organ.analyze(patient)

print(organ_output)