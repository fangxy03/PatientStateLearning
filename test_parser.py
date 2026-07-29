from llm_agents.evidence_parser import EvidenceParser



agent_output="""

Circulation State:
Severe compromise


Evidence:

- Heart Rate: 130 bpm

- SBP: 80 mmHg

- SpO2: 91%


Confidence score:
4/5

"""



parser=EvidenceParser()


result=parser.parse(
    agent_output
)


print(result)