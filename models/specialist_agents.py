import torch
import torch.nn as nn


class SpecialistAgent(nn.Module):

    def __init__(
        self,
        input_dim=256,
        output_dim=128
    ):
        super().__init__()


        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                output_dim
            ),

            nn.ReLU()
        )


        self.confidence = nn.Sequential(

            nn.Linear(
                output_dim,
                1
            ),

            nn.Sigmoid()

        )


    def forward(self,x):

        state=self.encoder(x)

        confidence=self.confidence(state)

        return state, confidence



# ==========================
# 三个专家Agent
# ==========================


class CirculationAgent(SpecialistAgent):

    def __init__(self):

        super().__init__()

        self.role = """
        You are a cardiovascular emergency specialist.

        Focus on:
        - blood pressure
        - heart rate
        - oxygen perfusion
        - shock
        - circulation failure

        Provide:
        circulation state
        evidence
        confidence
        """



class InfectionAgent(SpecialistAgent):

    def __init__(self):

        super().__init__()

        self.role = """
        You are an infection and sepsis specialist.

        Focus on:
        - temperature
        - inflammatory signs
        - infection risk
        - sepsis progression

        Provide:
        infection state
        evidence
        confidence
        """



class OrganAgent(SpecialistAgent):

    def __init__(self):

        super().__init__()

        self.role = """
        You are an organ dysfunction specialist.

        Focus on:
        - oxygen saturation
        - respiratory rate
        - organ failure
        - physiological instability

        Provide:
        organ state
        evidence
        confidence
        """