import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        sentences = positive + negative
        unique_words = sorted({word for sentence in sentences for word in sentence.split()})
        words_to_token = {word: token+1 for token,word in enumerate(unique_words)}
        # 2. Encode each sentence by replacing words with their IDs
        positive_encoded = [torch.Tensor([words_to_token[word] for word in sentence.split()]) for sentence in positive]
        negative_encoded = [torch.Tensor([words_to_token[word] for word in sentence.split()]) for sentence in negative]
        # 3. Combine positive + negative into one list of tensors
        encoded = positive_encoded + negative_encoded
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0.0)
