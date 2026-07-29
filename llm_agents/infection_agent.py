from .base_agent import LLM_Agent



class InfectionAgent(LLM_Agent):


    def __init__(self,llm):


        role="""

You are an infectious disease specialist.

Your task is to evaluate infection risk.

Focus on:

1. Temperature
2. Infection symptoms
3. Inflammatory signs


Do not analyze circulation.


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



Temperature:

{patient['temperature']}



Other information:

{patient}



Output:


1. Infection state

2. Evidence

3. Confidence score


"""


        return self.generate(prompt)