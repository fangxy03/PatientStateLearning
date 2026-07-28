import torch
import torch.nn as nn


class CirculationAgent(nn.Module):

    def __init__(
        self,
        input_dim=256,
        state_dim=128
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),


            nn.Linear(
                256,
                state_dim
            ),

            nn.ReLU()

        )


    def forward(self,x):

        circulation_state = self.network(x)

        return circulation_state