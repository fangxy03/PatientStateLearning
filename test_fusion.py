import torch

from models.multimodal_fusion import MultiModalFusion



# 模拟Patient Encoder输出

patient_state=torch.randn(
    4,
    256
)


# 模拟Qwen输出

text_state=torch.randn(
    4,
    5120
)



fusion=MultiModalFusion()


shared_state=fusion(
    patient_state,
    text_state
)


print(
    shared_state.shape
)