import torch
import torch.nn as nn


class ReliabilityEstimator(nn.Module):

    def __init__(self,dim=128):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(dim,64),

            nn.ReLU(),

            nn.Linear(64,1),

            nn.Sigmoid()

        )


    def forward(self,x):

        return self.net(x)