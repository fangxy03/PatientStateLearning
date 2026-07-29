from llm_agents.qwen_model import QwenModel

from llm_agents.circulation_agent import CirculationAgent
from llm_agents.infection_agent import InfectionAgent
from llm_agents.organ_agent import OrganAgent
from llm_agents.evidence_parser import EvidenceParser


print("================")
print("Loading agents")
print("================")


# 只加载一次Qwen

llm = QwenModel()



circ = CirculationAgent(llm)

inf = InfectionAgent(llm)

organ = OrganAgent(llm)

parser = EvidenceParser()

patient={


"age":72,


"chiefcomplaint":
"Chest pain",


"heartrate":130,


"sbp":80,


"o2sat":91,


"temperature":38.5


}




print("\n=====Circulation=====")


circ_output = circ.analyze(
    patient
)


print("\nRaw Output:")

print(circ_output)



circ_evidence = parser.parse(
    circ_output
)


print("\nParsed Evidence:")

print(circ_evidence)


print("\n=====Infection=====")


inf_output = inf.analyze(
    patient
)


print("\nRaw Output:")

print(inf_output)



inf_evidence = parser.parse(
    inf_output
)


print("\nParsed Evidence:")

print(inf_evidence)


print("\n=====Organ=====")


organ_output = organ.analyze(
    patient
)


print("\nRaw Output:")

print(organ_output)



organ_evidence = parser.parse(
    organ_output
)


print("\nParsed Evidence:")

print(organ_evidence)