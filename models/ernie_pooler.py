import torch
import torch.nn as nn
        
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Ernie
class ErniePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # ----- Fused dense + activation -----
        return self.activation(self.dense(hidden_states[:, 0]))

        # Original implementation (commented out for reference):
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)
        