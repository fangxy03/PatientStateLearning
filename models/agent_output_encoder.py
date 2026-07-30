import torch
import torch.nn as nn



class AgentOutputEncoder(nn.Module):


    def __init__(self):

        super().__init__()


        self.projection = nn.Sequential(

            nn.Linear(
                5120,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                128
            )

        )



    def forward(self,x):


        target_dtype = next(self.projection.parameters()).dtype

        if x.dtype != target_dtype:

            x = x.to(dtype=target_dtype)


        return self.projection(x)