import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = torch.Tensor([torch.inf])
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_indices = torch.randint(len(data)-context_length, size=(batch_size,))
            X = torch.stack([data[start : start+context_length] for start in start_indices])
            Y = torch.stack([data[start + 1 : start + 1 + context_length] for start in start_indices])
            y_hat = model(X) # forward pass
            B, T, C = y_hat.shape
            loss = F.cross_entropy(y_hat.reshape(B * T, C), Y.reshape(B * T)) # loss comp
            optimizer.zero_grad()
            loss.backward() # backward pass
            optimizer.step()
        
        return round(loss.item(), 4)
