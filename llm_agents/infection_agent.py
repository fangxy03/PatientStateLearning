from .base_agent import LLM_Agent



class InfectionAgent(LLM_Agent):


    def __init__(self):

        super().__init__()



    def analyze(self,patient):


        prompt=f"""

You are an infectious disease specialist.


Analyze infection risk.


Focus on:

- temperature
- fever
- inflammatory signs
- sepsis risk


Patient:

{patient}



Output:

Infection State:

Evidence:

Confidence score:

"""


        return self.generate(prompt)