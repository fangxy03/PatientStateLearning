from .base_agent import LLM_Agent



class OrganAgent(LLM_Agent):


    def __init__(self):

        super().__init__()



    def analyze(self,patient):


        prompt=f"""

You are an organ dysfunction specialist.


Analyze physiological organ status.


Focus on:

- oxygen saturation
- respiratory rate
- respiratory failure
- organ dysfunction


Patient:

{patient}



Output:

Organ State:

Evidence:

Confidence score:

"""


        return self.generate(prompt)