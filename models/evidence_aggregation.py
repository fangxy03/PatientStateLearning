import torch
import torch.nn as nn


class EvidenceAggregation(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,x):

        """
        x:
        [batch, agents, dim]

        example:
        [4,3,128]

        """

        # 三个agent融合

        state = torch.mean(
            x,
            dim=1
        )


        return state