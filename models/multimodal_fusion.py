import torch.nn as nn


class MultimodalFusion(nn.Module):

    def __init__(self):

        super().__init__()


        self.fc=nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2)

        )


    def forward(
        self,
        patient_state,
        text_state
    ):


        x=torch.cat(
            [
                patient_state,
                text_state
            ],
            dim=1
        )


        return self.fc(x)