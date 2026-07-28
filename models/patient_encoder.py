import torch
import torch.nn as nn


class PatientEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        state_dim=256
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                512,
                state_dim
            ),

            nn.ReLU()

        )


    def forward(self,x):

        state = self.encoder(x)

        return state