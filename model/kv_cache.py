import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.hstack((self.cache_k, new_k)) # new_k should be batch_size, new_seq_len, model_dim
            self.cache_v = torch.hstack((self.cache_v, new_v)) # new_v should be batch_size, new_seq_len, model_dim
        print("K cache shape: ", self.cache_k.shape)
        print("V cache shape: ", self.cache_v.shape)
        return (self.cache_k, self.cache_v)

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.model_dim = model_dim

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        print("x shape: ", x.shape)
        Q = self.q_proj(x) # batch_size, model_dim, model_dim
        K = self.k_proj(x) # batch_size, model_dim, model_dim
        V = self.v_proj(x) # batch_size, model_dim, model_dim
        print("Q shape: ", Q.shape)
        print("K shape: ", K.shape)
        print("V shape: ", V.shape)

        if kv_cache is None:
            kv_cache = KVCache()
        cache_k, cache_v = kv_cache.update(new_k=K, new_v=V)

        print(Q @ torch.transpose(cache_k, 1, 2))
        attention_score = nn.functional.softmax(Q @ torch.transpose(cache_k, 1, 2) / (self.model_dim**0.5), dim=-1)
        attention = attention_score @ cache_v

        return (torch.round(attention, decimals=4), kv_cache)
