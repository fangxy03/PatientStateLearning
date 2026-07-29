from .base_agent import LLM_Agent



class OrganAgent(LLM_Agent):


    def __init__(self,llm):


        role="""

You are an intensive care specialist.


Your task is to evaluate organ dysfunction.


Focus on:

1. Oxygenation

2. Respiratory failure

3. Organ impairment


"""


        super().__init__(
            llm,
            role
        )



    def analyze(self,patient):


        prompt=f"""

Patient information:


{patient}



Evaluate:


1. Organ dysfunction state

2. Evidence

3. Confidence score


"""


        return self.generate(prompt)