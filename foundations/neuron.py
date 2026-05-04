import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        y_hat = w @ x + b
        if activation == "sigmoid":
            y_hat = 1/(1 + np.exp(-y_hat))
        elif activation == "relu":
            y_hat = max(0.0, y_hat)
        else:
            return math.nan
        return round(y_hat, 5)
