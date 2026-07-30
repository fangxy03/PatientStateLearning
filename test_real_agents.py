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



device="cuda"



print("================")
print("Loading Dataset")
print("================")


dataset=ClinicalDataset(

    "data/subset/patient_subset.csv"

)


print(
"Dataset size:",
len(dataset)
)



patient,label=dataset[0]


print("\nPatient:")
print(patient)


print("\nLabel:")
print(label)



# =====================
# LLM
# =====================


print("================")
print("Loading LLM")
print("================")


llm=QwenModel()



circ=CirculationAgent(llm)

inf=InfectionAgent(llm)

organ=OrganAgent(llm)



# =====================
# Agent reasoning
# =====================


print("\n=====Circulation=====")

circ_text=circ.analyze(patient)

print(circ_text)



print("\n=====Infection=====")

inf_text=inf.analyze(patient)

print(inf_text)



print("\n=====Organ=====")

organ_text=organ.analyze(patient)

print(organ_text)



# =====================
# Agent output encoding
# =====================


print("================")
print("Encoding Agent outputs")
print("================")



encoder=AgentOutputEncoder().to(device)



texts=[

circ_text,

inf_text,

organ_text

]



states=[]



for text in texts:


    # Qwen embedding

    hidden=llm.encode(text)


    hidden=hidden.to(device)


    vector=encoder(hidden)


    states.append(vector)



# [3,1,128]

agent_states=torch.stack(states)



print(
"Agent states:",
agent_states.shape
)



# [1,3,128]

agent_states=agent_states.permute(
    1,0,2
)



print(
"Fusion input:",
agent_states.shape
)



# =====================
# Evidence Fusion
# =====================


interaction=EvidenceInteraction().to(device)


aggregation=EvidenceAggregation().to(device)


classifier=ESIClassifier().to(device)



confidence=torch.tensor(

[
[
0.8,
0.8,
0.8
]
],

dtype=torch.float32

).to(device)



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

"Ground truth:",

label.item()+1

)