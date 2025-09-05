import torch
import torch.nn as nn
from plugins.fused_layernorm import FusedDenseLayerNorm

# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Ernie
class ErnieOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # must be intermediate_size -> hidden_size
        self.fused_dense_ln = FusedDenseLayerNorm(
          config.intermediate_size, 
          config.hidden_size, 
          eps=config.layer_norm_eps)

        # Original implementation (commented out for reference):
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # ----- Fused dense + layernorm -----
        return self.fused_dense_ln(hidden_states, residual=input_tensor)

        # Original implementation:
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # return hidden_states

        