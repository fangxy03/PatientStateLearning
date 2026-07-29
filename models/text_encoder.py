import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ClinicalTextEncoder(nn.Module):

    def __init__(
        self,
        model_path,
        hidden_dim=768
    ):

        super().__init__()


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )


        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )


        self.hidden_dim = hidden_dim



    def forward(self, texts):


        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )


        inputs = {
            k:v.to(self.model.device)
            for k,v in inputs.items()
        }


        outputs = self.model(
            **inputs
        )


        # 取最后一层平均池化

        hidden_states = outputs.last_hidden_state


        text_feature = hidden_states.mean(
            dim=1
        )


        return text_feature