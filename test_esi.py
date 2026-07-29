import torch

from models.esi_classifier import ESIClassifier


model = ESIClassifier()


x=torch.randn(
    4,
    512
)


logits,state=model(x)


print(
    "ESI logits:",
    logits.shape
)


print(
    "Clinical state:",
    state.shape
)