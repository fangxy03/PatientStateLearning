from .base_agent import LLM_Agent



class CirculationAgent(LLM_Agent):


    def __init__(self,llm):


        role = """

You are a cardiovascular emergency specialist.

Your task is to evaluate patient circulation state.

Focus on:

1. Blood pressure
2. Heart rate
3. Shock signs
4. Perfusion

Do not analyze infection or organ dysfunction.


"""


        super().__init__(
            llm,
            role
        )



    def analyze(self,patient):


        prompt=f"""

Patient information:


Age:
{patient['age']}


Chief complaint:

{patient['chiefcomplaint']}



Vital signs:


Heart rate:

{patient['heartrate']}


SBP:

{patient['sbp']}


SpO2:

{patient['o2sat']}



Please output:


1. Circulation state

2. Evidence

3. Confidence score


"""


        return self.generate(prompt)