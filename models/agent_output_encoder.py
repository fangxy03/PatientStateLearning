import torch
import torch.nn as nn


class AgentOutputEncoder(nn.Module):

    def __init__(
        self,
        qwen_dim=5120,
        output_dim=512
    ):
        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                qwen_dim,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                output_dim
            )
        )


    def forward(self,x):

        return self.projection(x)