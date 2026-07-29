class LLM_Agent:



    def __init__(
        self,
        llm,
        role
    ):


        self.llm = llm

        self.role = role



    def generate(self,prompt):


        return self.llm.generate(
            self.role,
            prompt
        )