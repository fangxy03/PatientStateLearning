import torch
import torch.nn as nn



class MultiModalFusion(nn.Module):

    def __init__(
        self,
        patient_dim=256,
        text_dim=5120,
        hidden_dim=512
    ):

        super().__init__()


        self.fc = nn.Sequential(

            nn.Linear(
                patient_dim + text_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU()
        )



    def forward(
        self,
        patient_state,
        text_state
    ):


        # 拼接结构化状态和文本状态

        fusion_input = torch.cat(
            [
                patient_state,
                text_state
            ],
            dim=1
        )


        # 得到共享患者状态

        shared_state = self.fc(
            fusion_input
        )


        return shared_state