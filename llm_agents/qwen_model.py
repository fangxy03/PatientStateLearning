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


        self.device = self.model.device


        print("Qwen loaded on GPU")



    def generate(self, system_prompt, user_prompt):

        """
        给Agent调用
        输入:
        system_prompt
        user_prompt

        输出:
        LLM文本
        """


        messages=[

            {
                "role":"system",
                "content":system_prompt
            },

            {
                "role":"user",
                "content":user_prompt
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



        with torch.no_grad():

            outputs=self.model.generate(

                **inputs,

                max_new_tokens=512,

                do_sample=False

            )



        response=self.tokenizer.decode(

            outputs[0][inputs.input_ids.shape[-1]:],

            skip_special_tokens=True

        )


        return response



    def encode(self,text):

        """
        后面Agent output encoder需要
        """

        inputs=self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=512

        ).to(self.model.device)



        with torch.no_grad():

            outputs=self.model(

                **inputs,

                output_hidden_states=True

            )


        # 最后一层hidden state

        hidden=outputs.hidden_states[-1]


        # mean pooling

        embedding=hidden.mean(dim=1)


        return embedding