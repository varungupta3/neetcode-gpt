import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        eps = 1e-5
        npx = np.array(x)
        npx_rms = np.sqrt(np.mean(npx * npx) + eps)
        x_hat = np.round(np.array(gamma) * npx / npx_rms, 4)
        return x_hat.tolist()
