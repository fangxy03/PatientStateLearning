import torch
import torch.nn as nn



class EvidenceFusion(nn.Module):


    def __init__(self):

        super().__init__()



    def forward(
        self,
        states,
        reliability
    ):


        weighted = (
            states *
            reliability.unsqueeze(-1)
        )


        shared_state = weighted.sum(
            dim=1
        )


        return shared_state