from circulation_agent import CirculationAgent
from infection_agent import InfectionAgent
from organ_agent import OrganAgent



patient="""

Age:72

Chief complaint:
Chest pain


Vital signs:

HR:130

SBP:80

SpO2:91


Temperature:
38.5


Respiratory rate:
32

"""



print("================")
print("Loading agents")
print("================")


circ=CirculationAgent()

inf=InfectionAgent()

organ=OrganAgent()



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