import torch

from models.evidence_interaction import EvidenceInteraction



model = EvidenceInteraction()



# 三个agent embedding

agent_states=torch.randn(
    4,
    3,
    128
)



confidence=torch.tensor(
[
[0.8,0.7,0.9],
[0.9,0.8,0.7],
[0.6,0.9,0.8],
[0.8,0.8,0.8]
]
)



out=model(
    agent_states,
    confidence
)


print(out.shape)