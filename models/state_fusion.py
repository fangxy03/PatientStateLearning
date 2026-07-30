import torch
import torch.nn as nn


class StateFusion(nn.Module):

    def __init__(
        self,
        input_dim=128,
        output_dim=512
    ):
        super().__init__()

        self.fusion = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                output_dim
            )

        )


    def forward(self,x):

        return self.fusion(x)