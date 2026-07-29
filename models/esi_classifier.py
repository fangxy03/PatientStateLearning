import torch
import torch.nn as nn


class ESIClassifier(nn.Module):

    def __init__(
        self,
        input_dim=512,
        state_dim=128,
        num_classes=5
    ):

        super().__init__()


        self.state_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                state_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )


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

        clinical_state = self.state_projection(x)

        logits = self.classifier(
            clinical_state
        )

        return logits, clinical_state