import torch
import torch.nn as nn


class ESIClassifier(nn.Module):

    def __init__(
        self,
        state_dim=128,
        num_classes=5
    ):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                state_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self,x):

        return self.classifier(x)