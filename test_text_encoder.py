from models.text_encoder import ClinicalTextEncoder


model_path="/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"



encoder=ClinicalTextEncoder(
    model_path
)


text=[
"""
72 year old patient.
Chief complaint:
Chest pain.
Hypotension.
Tachycardia.
"""
]


state=encoder(text)


print(state.shape)