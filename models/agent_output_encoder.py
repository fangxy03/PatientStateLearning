import torch
import torch.nn as nn


class AgentOutputEncoder(nn.Module):

    def __init__(
        self,
        input_dim=5120,
        output_dim=128
    ):

        super().__init__()


        self.projection = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                output_dim
            )

        )


    def forward(self,x):

        return self.projection(x)