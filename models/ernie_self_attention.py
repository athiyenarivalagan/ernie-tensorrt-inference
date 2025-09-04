import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
# from transformers.utils import deprecate_kwarg
from transformers.cache_utils import Cache, EncoderDecoderCache


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Ernie
class ErnieSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads                           # 12 (number of heads)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768 / 12 = 64 (head dimension)
        self.all_head_size = self.num_attention_heads * self.attention_head_size        # 12 * 64 = 768 (embedding dimension)

        # ----- Fused the Q, K, V layers -----
        # self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size) # (768, 3 * 768)
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size)
        
        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # ----- Init the scaling factor -----
        self.inv_scale = 1.0 / math.sqrt(self.attention_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        # ----- Minimal code simplification to avoid repetition -----
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.layer_idx = layer_idx

    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Cache] = None, # accepts old keyword
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape 
        # query_layer = self.query(hidden_states)
        # query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
        #     1, 2
        # )

        # ----- Unify the variable internally -----
        pkv = past_key_value if past_key_value is not None else past_key_values
        
        # before the cache logic
        is_cross_attention = encoder_hidden_states is not None
        is_updated = False # ensure defined

        # ----- Removed intermediate allocations/copies -----
        if pkv is not None:
            if isinstance(pkv, EncoderDecoderCache):
                is_updated = pkv.is_updated.get(self.layer_idx)
                curr_past_key_value = pkv.cross_attention_cache if is_cross_attention else pkv.self_attention_cache
            else:
                curr_past_key_value = pkv

            # ----- Each step creates new "curr_past_key_values" -----
            # if past_key_values is not None:
            # if isinstance(past_key_values, EncoderDecoderCache):
            #     is_updated = past_key_values.is_updated.get(self.layer_idx)
            #     if is_cross_attention:
            #         # after the first generated id, we can subsequently re-use all key/value_layer from cache
            #         curr_past_key_value = past_key_values.cross_attention_cache
            #     else:
            #         curr_past_key_value = past_key_values.self_attention_cache
            # else:
            #     curr_past_key_value = past_key_values

        # ----- QKV computation -----
        # current_states = encoder_hidden_states if is_cross_attention else hidden_states (moved to the else statement)
        if is_cross_attention and pkv is not None and is_updated:
            # reuse k,v: only compute q for current hidden_states
            fused_q = self.qkv(hidden_states)
            # ----- revisit this section -----
            query_layer, _, _ = fused_q.chunk(3, dim=-1)
            query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)\
                             .transpose(1, 2) # [B,H,L,D]
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            current_states = encoder_hidden_states if is_cross_attention else hidden_states
            # ----- One fused GEMM for q, k, v -----
            fused_qkv= self.qkv(current_states) # output shape: (batch, seq_len, 3 * heads * head_dim) => (batch, seq_len, 2304)
            # ----- Reshape and Permute before splitting -----
            fused_qkv = fused_qkv.view(batch_size, seq_length, 3, self.num_attention_heads, self.attention_head_size)\
                .permute(2, 0, 3, 1, 4) # shape: (3, batch, heads, seq, head_dim) 
            # ----- Split into Q, K, V for all heads -----
            query_layer, key_layer, value_layer = fused_qkv[0], fused_qkv[1], fused_qkv[2]
            
            # key_layer = self.key(current_states)
            # value_layer = self.value(current_states)
            # key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
            # value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

            if pkv is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    pkv.is_updated[self.layer_idx] = True

        # ----- Scaled Dot-Product Attention -----
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.inv_scale
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # ----- Relative Position Embedding (could make more improvements) -----
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype) # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key


        # ----- The following op is already performed -----
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # ----- Fused Add + Softmax + Mixed Precision -----
        # Do you add attention_mask before converting to float32 or after?
        if attention_mask is not None:
            attention_probs = F.softmax((attention_scores.to(torch.float32) + attention_mask), dim=-1)\
                              .to(attention_scores.dtype)
        else:
            attention_probs = F.softmax(attention_scores.to(torch.float32), dim=-1)\
                              .to(attention_scores.dtype)

        # if attention_mask is not None:
        #     # Apply the attention mask is (precomputed for all layers in ErnieModel forward() function)
        #     attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
 
        # ----- context_layer: [B, H, L, D] -> [B, L, HD] -----
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, seq_length, self.all_head_size)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs