from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_path="/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)


print("Model loaded")
