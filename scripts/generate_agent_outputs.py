import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.clinical_dataset import ClinicalDataset


from llm_agents.qwen_model import QwenModel

from llm_agents.circulation_agent import CirculationAgent
from llm_agents.infection_agent import InfectionAgent
from llm_agents.organ_agent import OrganAgent



# =========================
# 配置
# =========================


DATA_PATH = str(REPO_ROOT / "data/subset/patient_subset.csv")

OUTPUT_PATH = str(REPO_ROOT / "data/agent_outputs.json")


# 测试数量
# None表示全部

MAX_SAMPLES = 1000



# =========================
# Dataset
# =========================


print("================")
print("Loading Dataset")
print("================")


dataset = ClinicalDataset(
    DATA_PATH
)



print(
    "Dataset size:",
    len(dataset)
)



# =========================
# Load Qwen
# =========================


print("================")
print("Loading Qwen")
print("================")


llm = QwenModel()



# 三个agent


circulation_agent = CirculationAgent(
    llm
)


infection_agent = InfectionAgent(
    llm
)


organ_agent = OrganAgent(
    llm
)



# =========================
# Generate
# =========================


results=[]



num=len(dataset)


if MAX_SAMPLES:

    num=min(
        num,
        MAX_SAMPLES
    )



print("================")
print("Generating Agent Outputs")
print("================")



for i in tqdm(range(num)):



    patient,label = dataset[i]



    try:


        circ = circulation_agent.analyze(
            patient
        )


        inf = infection_agent.analyze(
            patient
        )


        organ = organ_agent.analyze(
            patient
        )



        item={


            "index":i,


            "patient":patient,



            "circulation":circ,


            "infection":inf,


            "organ":organ,



            "label":
            int(label.item()+1)

        }



        results.append(
            item
        )



    except Exception as e:


        print(
            "Error:",
            i,
            e
        )


        continue





# =========================
# Save
# =========================



os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)



with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:


    json.dump(

        results,

        f,

        ensure_ascii=False,

        indent=2

    )




print("================")
print("Finished")
print("================")


print(
    "Saved:",
    OUTPUT_PATH
)


print(
    "Samples:",
    len(results)
)