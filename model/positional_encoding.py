import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pos_enc = np.arange(seq_len).reshape(-1, 1) # shape (seq_len, 1)
        dim_enc = np.arange(d_model).reshape(1, -1) # shape (1, d_model)
        PE = np.zeros((seq_len, d_model))
        PE[:, 0::2] = np.sin(pos_enc / 10000**(dim_enc[:,0::2]/d_model))
        PE[:, 1::2] = np.cos(pos_enc / 10000**(dim_enc[:,0:-1:2]/d_model))
        return np.round(PE, 5)
