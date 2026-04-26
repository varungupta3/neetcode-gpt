import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.key_layer(embedded) # (B, T, attention_dim)
        Q = self.query_layer(embedded) # (B, T, attention_dim)
        V = self.value_layer(embedded) # (B, T, attention_dim)
        
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = (Q @ torch.transpose(K, 1, 2) / (self.attention_dim ** 0.5)) # (B, T, T)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        lower_tri = torch.tril(torch.ones(scores.shape))
        mask = lower_tri == 0
        masked_scores = scores.masked_fill(mask, -torch.inf) # (B, T, T)
        
        # 4. Apply softmax(dim=2) to masked scores
        scores = nn.functional.softmax(masked_scores, dim=2)

        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(scores @ V, decimals=4) # (B, T, attention_dim)
