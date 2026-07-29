import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)



MODEL_PATH="/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"



class LLM_Agent:


    def __init__(self):

        print("Loading Qwen model...")


        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )


        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )


        self.model.eval()



    def generate(self,prompt):


        messages=[
            {
                "role":"system",
                "content":
                "You are a medical AI assistant."
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
        ).to(
            self.model.device
        )


        with torch.no_grad():

            outputs=self.model.generate(

                **inputs,

                max_new_tokens=512,

                temperature=0.2,

                do_sample=False

            )


        response=self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )


        return response