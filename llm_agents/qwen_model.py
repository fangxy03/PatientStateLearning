from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenModel:


    def __init__(self):

        print("Loading shared Qwen model...")


        model_path = (
        "/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"
        )


        self.device="cuda"


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


        self.model.eval()


        print("Qwen loaded on GPU")



    # ============================
    # 给Agent调用
    # ============================

    def generate(self,prompt):


        messages=[

            {
                "role":"user",
                "content":prompt
            }

        ]


        inputs=self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)



        with torch.no_grad():

            outputs=self.model.generate(

                inputs,

                max_new_tokens=512,

                do_sample=False

            )


        text=self.tokenizer.decode(

            outputs[0][inputs.shape[-1]:],

            skip_special_tokens=True

        )


        return text



    # ============================
    # 给Agent Encoder调用
    # ============================

    def encode(self,text):


        inputs=self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=512

        ).to(self.device)



        with torch.no_grad():

            outputs=self.model.model.embed_tokens(
                inputs.input_ids
            )


        # mean pooling

        hidden=outputs.mean(
            dim=1
        )


        return hidden