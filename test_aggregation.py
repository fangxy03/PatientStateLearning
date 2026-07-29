import torch

from models.evidence_aggregation import EvidenceAggregation


model=EvidenceAggregation()


x=torch.randn(
    4,
    3,
    128
)


out=model(x)


print(out.shape)