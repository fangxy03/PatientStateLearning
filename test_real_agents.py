from dataset.clinical_dataset import ClinicalDataset


from llm_agents.qwen_model import QwenModel

from llm_agents.circulation_agent import CirculationAgent
from llm_agents.infection_agent import InfectionAgent
from llm_agents.organ_agent import OrganAgent



from models.agent_output_encoder import AgentOutputEncoder

from models.evidence_interaction import EvidenceInteraction
from models.evidence_aggregation import EvidenceAggregation
from models.esi_classifier import ESIClassifier


import torch



print("================")
print("Loading Dataset")
print("================")


dataset = ClinicalDataset(
    "data/subset/patient_subset.csv"
)


print(
    "Dataset size:",
    len(dataset)
)



# 一个病人测试

patient,label = dataset[0]


print("\nPatient:")
print(patient)


print("\nESI label:")
print(label)



# ==========================
# Load LLM
# ==========================


print("================")
print("Loading LLM")
print("================")


llm = QwenModel()



# 三个agent

circ = CirculationAgent(llm)

inf = InfectionAgent(llm)

organ = OrganAgent(llm)



# ==========================
# Agent reasoning
# ==========================


print("\n===== Circulation =====")


circ_output = circ.analyze(patient)

print(circ_output)



print("\n===== Infection =====")


inf_output = inf.analyze(patient)

print(inf_output)



print("\n===== Organ =====")


organ_output = organ.analyze(patient)

print(organ_output)



# ==========================
# Agent output encoding
# ==========================


print("\n================")
print("Encoding Agent outputs")
print("================")



encoder = AgentOutputEncoder()



agent_texts=[

    circ_output,

    inf_output,

    organ_output

]



agent_vectors=[]



for text in agent_texts:


    # Qwen text embedding

    hidden = llm.encode(text)


    vector = encoder(hidden)


    agent_vectors.append(vector)



# [3,1,128]

agent_states=torch.stack(
    agent_vectors
)


print(
    "Agent states:",
    agent_states.shape
)



# 改成fusion需要的格式

# [batch,agent,dim]


agent_states=agent_states.permute(
    1,
    0,
    2
)



print(
    "Fusion input:",
    agent_states.shape
)



# ==========================
# Evidence Fusion
# ==========================


print("\n================")
print("Evidence Fusion")
print("================")



interaction = EvidenceInteraction()


aggregation = EvidenceAggregation()


classifier = ESIClassifier()



# confidence来自agent解析

confidence=torch.tensor(

    [[
        circ_output["confidence"],
        inf_output["confidence"],
        organ_output["confidence"]
    ]]

)



print(
    "Confidence:",
    confidence
)



x=interaction(

    agent_states,

    confidence

)


print(
    "After interaction:",
    x.shape
)



x=aggregation(x)


print(
    "After aggregation:",
    x.shape
)



logits=classifier(x)



print(
    "ESI logits:",
    logits
)



prediction=torch.argmax(
    logits,
    dim=1
)


print(
    "Prediction ESI:",
    prediction.item()+1
)


print(
    "Ground Truth ESI:",
    label.item()+1
)