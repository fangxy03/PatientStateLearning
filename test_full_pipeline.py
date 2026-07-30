import torch


from models.evidence_interaction import EvidenceInteraction
from models.evidence_aggregation import EvidenceAggregation
from models.esi_classifier import ESIClassifier



# ==========================
# 模拟三个agent输出
# ==========================

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



# ==========================
# 初始化模块
# ==========================

interaction=EvidenceInteraction()

fusion=EvidenceAggregation()

classifier=ESIClassifier()



# ==========================
# Pipeline
# ==========================

x = interaction(
    agent_states,
    confidence
)


print(
"after interaction:",
x.shape
)



x = fusion(x)


print(
"after aggregation:",
x.shape
)



logits = classifier(x)


print(
"ESI output:",
logits.shape
)