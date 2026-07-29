import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_path,
        output_dim=128
    ):

        super().__init__()


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )


        self.model = AutoModel.from_pretrained(
            model_path
        )


        hidden_dim = self.model.config.hidden_size


        self.projector = nn.Sequential(

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.ReLU()

        )



    def forward(
        self,
        texts
    ):

        """
        texts:
        [
          "patient circulation shock...",
          "infection risk...",
          "organ dysfunction..."
        ]
        """


        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


        device = next(
            self.model.parameters()
        ).device


        tokens = {
            k:v.to(device)
            for k,v in tokens.items()
        }



        outputs = self.model(
            **tokens
        )


        # CLS token

        embedding = outputs.last_hidden_state[:,0,:]


        embedding = self.projector(
            embedding
        )


        return embedding