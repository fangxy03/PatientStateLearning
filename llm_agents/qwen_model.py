from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenModel:


    def __init__(self):

        print("Loading shared Qwen model...")


        model_path = (
        "/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )


        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )


    def generate(self,system,prompt):


        messages=[

        {
        "role":"system",
        "content":system
        },

        {
        "role":"user",
        "content":prompt
        }

        ]


        text=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        inputs=self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)


        outputs=self.model.generate(
            **inputs,
            max_new_tokens=300
        )


        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )