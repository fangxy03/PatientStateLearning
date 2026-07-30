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



# 保存三个agent结果

agent_outputs = [

    circ_output,

    inf_output,

    organ_output

]

import torch

from models.agent_output_encoder import AgentOutputEncoder



print("\n================")
print("Encoding Agent outputs")
print("================")



encoder = AgentOutputEncoder()



agent_vectors=[]



for output in agent_outputs:


    # 暂时模拟Qwen文本embedding

    hidden = torch.randn(
        1,
        5120
    )


    vector = encoder(hidden)


    agent_vectors.append(vector)



agent_states = torch.stack(
    agent_vectors
)



print(
    "Agent states:",
    agent_states.shape
)

# 调整维度给Evidence Fusion

agent_states = agent_states.permute(
    1,
    0,
    2
)


print(
    "Fusion input:",
    agent_states.shape
)

from models.evidence_interaction import EvidenceInteraction
from models.evidence_aggregation import EvidenceAggregation
from models.esi_classifier import ESIClassifier

print("\n================")
print("Evidence Fusion")
print("================")


interaction = EvidenceInteraction()

aggregation = EvidenceAggregation()

classifier = ESIClassifier()



# 三个agent confidence

confidence=torch.tensor(
    [
        [
            0.8,
            0.7,
            0.9
        ]
    ]
)



x = interaction(
    agent_states,
    confidence
)


print(
    "After interaction:",
    x.shape
)



x = aggregation(x)


print(
    "After aggregation:",
    x.shape
)



logits = classifier(x)



print(
    "ESI prediction:",
    logits.shape
)