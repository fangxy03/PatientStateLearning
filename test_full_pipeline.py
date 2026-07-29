import torch


from models.evidence_interaction import EvidenceInteraction

from models.evidence_aggregation import EvidenceAggregation

from models.esi_classifier import ESIClassifier



# 模拟三个agent输出

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



interaction=EvidenceInteraction()


fusion=EvidenceAggregation()


classifier=ESIClassifier()



x=interaction(
    agent_states,
    confidence
)



x=fusion(x)



logits=classifier(x)



print(
"ESI output:",
logits.shape
)