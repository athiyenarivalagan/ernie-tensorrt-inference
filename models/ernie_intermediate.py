import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Ernie
class ErnieIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # ----- Simplified if-else statement -----
        self.intermediate_act_fn=ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

        # Original implementation (commented out for reference):
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ----- Fused dense + activation -----
        return self.intermediate_act_fn(self.dense(hidden_states))
        # Original implementation:
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        # return hidden_staes