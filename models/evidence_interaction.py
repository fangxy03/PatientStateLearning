import torch
import torch.nn as nn


class EvidenceInteraction(nn.Module):

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True
        )


        self.norm = nn.LayerNorm(
            input_dim
        )


    def forward(
        self,
        agent_states,
        confidence
    ):

        """
        agent_states:
        [B,3,128]

        confidence:
        [B,3]

        """


        # confidence作为可靠性权重

        reliability = confidence.unsqueeze(-1)


        weighted_state = (
            agent_states
            *
            reliability
        )


        # agent之间通信

        interaction,_ = self.attention(
            weighted_state,
            weighted_state,
            weighted_state
        )


        output = self.norm(
            interaction + weighted_state
        )


        return output