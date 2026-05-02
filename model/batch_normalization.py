import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        eps = 1e-5
        npx = np.array(x) # Batch, Features
        npgamma = np.array(gamma).reshape(1, -1) # Features
        npbeta = np.array(beta).reshape(1, -1) # Features
        nprunning_mean = np.array(running_mean).reshape(1, -1) # Features
        nprunning_var = np.array(running_var).reshape(1, -1) # Features
        if training:
            x_mean = np.mean(npx, axis=0) # Average over batch ==> Features
            x_var = np.var(npx, axis=0) # Variance over batch ==> Features
            x_hat = (npx - x_mean.reshape(1, -1)) / np.sqrt(x_var + eps)
            y = npgamma * x_hat + npbeta
            nprunning_mean = nprunning_mean * (1-momentum) + x_mean * momentum
            nprunning_var = nprunning_var * (1-momentum) + x_var * momentum
        else:
            x_hat = (npx - nprunning_mean) / np.sqrt(nprunning_var + eps)
            y = npgamma * x_hat + npbeta
        y_list = np.round(y, 4).tolist()
        running_mean_list = np.round(nprunning_mean.squeeze(), 4).tolist()
        running_var_list = np.round(nprunning_var.squeeze(), 4).tolist()
        return (y_list, running_mean_list, running_var_list)