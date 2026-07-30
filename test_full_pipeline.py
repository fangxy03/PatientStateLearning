import torch


from models.agent_state_encoder import AgentStateEncoder

from models.evidence_interaction import EvidenceInteraction

from models.evidence_aggregation import EvidenceAggregation

from models.esi_classifier import ESIClassifier



batch=4


encoder=AgentStateEncoder()



# 三个agent的LLM embedding

circ_text=torch.randn(
batch,
5120
)

inf_text=torch.randn(
batch,
5120
)

organ_text=torch.randn(
batch,
5120
)



circ_state=encoder(circ_text)

inf_state=encoder(inf_text)

organ_state=encoder(organ_text)



agent_states=torch.stack(
[
circ_state,
inf_state,
organ_state
],
dim=1
)



print(
"agent states:",
agent_states.shape
)



confidence=torch.tensor(
[
[0.8,0.7,0.9],
[0.9,0.8,0.7],
[0.6,0.9,0.8],
[0.8,0.8,0.8]
]
)



interaction=EvidenceInteraction()


fusion=EvidenceAggregation()


classifier=ESIClassifier()



x=interaction(
agent_states,
confidence
)


print(
"interaction:",
x.shape
)



x=fusion(x)


print(
"fusion:",
x.shape
)



logits=classifier(x)


print(
"ESI:",
logits.shape
)