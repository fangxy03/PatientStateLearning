import torch

from models.text_encoder import TextEncoder


model_path="/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"


encoder=TextEncoder(
    model_path
)



texts=[

"Patient has severe circulation shock. SBP 80 HR130",

"Patient has fever 38.5 and possible infection",

"Patient has hypoxia SpO2 91"

]


output=encoder(texts)


print(output.shape)