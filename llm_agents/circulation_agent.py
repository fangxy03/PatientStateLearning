from .base_agent import LLM_Agent



class CirculationAgent(LLM_Agent):


    def __init__(self):

        super().__init__()



    def analyze(self,patient):


        prompt=f"""

You are a cardiovascular emergency specialist.


Your role:
Analyze patient circulation status.


Focus on:

- heart rate
- systolic blood pressure
- diastolic blood pressure
- oxygen saturation
- shock signs


Patient information:

{patient}



Output strictly:

Circulation State:

Evidence:

Confidence score:


"""


        return self.generate(prompt)