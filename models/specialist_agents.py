import torch
import torch.nn as nn



class SpecialistAgent(nn.Module):

    def __init__(
        self,
        input_dim=256,
        state_dim=128
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                state_dim
            ),

            nn.ReLU()
        )


    def forward(self,x):

        state = self.net(x)

        return state



class CirculationAgent(SpecialistAgent):

    pass



class InfectionAgent(SpecialistAgent):

    pass



class OrganAgent(SpecialistAgent):

    pass