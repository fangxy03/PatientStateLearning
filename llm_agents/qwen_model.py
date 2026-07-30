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



    ##################################
    # 给Agent调用
    ##################################

    def generate(self, prompt):


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )


        inputs = {
            k:v.to(self.device)
            for k,v in inputs.items()
        }



        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )



        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        return text



    ##################################
    # Agent输出编码
    ##################################

    def encode(self,text):


        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        inputs = {
            k:v.to(self.device)
            for k,v in inputs.items()
        }


        with torch.no_grad():


            outputs=self.model(
                **inputs,
                output_hidden_states=True
            )


        hidden_states = outputs.hidden_states[-1]


        # mean pooling

        embedding = hidden_states.mean(
            dim=1
        )


        return embedding.cpu()