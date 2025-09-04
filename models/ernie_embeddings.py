import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ErnieEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.use_task_id = config.use_task_id
        if config.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
        # ----- Yet to implement the following modifications -----
        # self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False)
        # self.register_buffer("token_type_ids", torch.zeros(1, config.max_position_embeddings, dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # ----- Simplified the if-else statement -----
        input_shape = input_ids.size() if input_ids.size() is not None else inputs_embeds.size()[:-1]
        # if input_ids is not None:
        #     input_shape = input_ids.size()
        # else:
        #     input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(input_shape[0], seq_length)
            # if hasattr(self, "token_type_ids"):
                # buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                # token_type_ids = buffered_token_type_ids_expanded
            # else:
            #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # if inputs_embeds is None:
        #     inputs_embeds = self.word_embeddings(input_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids) 
        # embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            embeddings += self.position_embeddings(position_ids)
            # position_embeddings = self.position_embeddings(position_ids)
            # embeddings += position_embeddings

        # add `task_type_id` for ERNIE model
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            embeddings += self.task_type_embeddings(task_type_ids)
            # task_type_embeddings = self.task_type_embeddings(task_type_ids)
            # embeddings += task_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings