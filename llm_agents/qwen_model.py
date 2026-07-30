from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenModel:


    def __init__(self):

        print("Loading shared Qwen model...")


        model_path = (
        "/media/ubuntu/Student/fxy/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"
        )


        # tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )


        # 加载模型

        self.model = AutoModelForCausalLM.from_pretrained(

            model_path,

            torch_dtype=torch.float16,

            device_map={
                "":0
            },

            trust_remote_code=True
        )


        self.model.eval()


        print("Qwen loaded on GPU")



    def generate(self,prompt):


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )


        inputs = {
            k:v.cuda()
            for k,v in inputs.items()
        }



        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=512,

                do_sample=False,

                pad_token_id=self.tokenizer.eos_token_id

            )


        result = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )


        return result