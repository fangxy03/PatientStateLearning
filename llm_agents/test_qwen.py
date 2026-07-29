from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_path="/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


prompt="""

You are a cardiovascular emergency specialist.

Analyze this patient.

Age:72

Chief complaint:
Chest pain

Vital signs:

HR:130

SBP:80

SpO2:91


Output:

1. Patient circulation state
2. Evidence
3. Confidence score

"""


inputs=tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)


outputs=model.generate(
    **inputs,
    max_new_tokens=512
)


response=tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print(response)