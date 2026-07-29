from llm_agents.qwen_model import QwenModel

from llm_agents.circulation_agent import CirculationAgent
from llm_agents.infection_agent import InfectionAgent
from llm_agents.organ_agent import OrganAgent



print("================")
print("Loading agents")
print("================")


# 只加载一次Qwen

llm = QwenModel()



circ = CirculationAgent(llm)

inf = InfectionAgent(llm)

organ = OrganAgent(llm)



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

print(
circ.analyze(patient)
)



print("\n=====Infection=====")

print(
inf.analyze(patient)
)



print("\n=====Organ=====")

print(
organ.analyze(patient)
)