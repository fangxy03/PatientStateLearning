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



patient,label = dataset[0]



print("\nPatient:")
print(patient)


print("\nESI label:")
print(label)



# =====================
# Qwen
# =====================


print("================")
print("Loading LLM")
print("================")


llm=QwenModel()



# =====================
# Agents
# =====================


circ=CirculationAgent(llm)

inf=InfectionAgent(llm)

organ=OrganAgent(llm)



# =====================
# Agent reasoning
# =====================


print("\n===== Circulation =====")

circ_output=circ.analyze(patient)

print(circ_output)



print("\n===== Infection =====")

inf_output=inf.analyze(patient)

print(inf_output)



print("\n===== Organ =====")

organ_output=organ.analyze(patient)

print(organ_output)



# =================================
# Agent output -> text embedding
# =================================


print("\n================")
print("Encoding Agent outputs")
print("================")



encoder=AgentOutputEncoder()



agent_outputs=[

    circ_output,

    inf_output,

    organ_output

]



agent_vectors=[]



for output in agent_outputs:


    # 如果agent返回字符串

    if isinstance(output,str):

        text=output

    else:

        text=output["raw"]



    hidden=llm.encode(text)


    vector=encoder(hidden)


    agent_vectors.append(vector)



agent_states=torch.stack(
    agent_vectors
)



print(
    "Agent states:",
    agent_states.shape
)



# [agent,batch,dim]

agent_states=agent_states.permute(
    1,
    0,
    2
)



print(
    "Fusion input:",
    agent_states.shape
)



# ======================
# Evidence Fusion
# ======================


print("\n================")
print("Evidence Fusion")
print("================")



interaction=EvidenceInteraction()


aggregation=EvidenceAggregation()


classifier=ESIClassifier()



# 暂时从文本解析confidence失败

# 先固定

confidence=torch.tensor(

[
[
0.8,
0.8,
0.8
]

]

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