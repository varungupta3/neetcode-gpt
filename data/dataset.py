import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        words = raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        max_indice = max(len(words) - context_length, 0)
        start_indices = torch.randint(low=0, high=max_indice, size=(batch_size,))
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        context = [[word for word in words[i:i+context_length]] for i in start_indices]
        next_token = [[word for word in words[i+1:i+context_length+1]] for i in start_indices]
        return (context, next_token)