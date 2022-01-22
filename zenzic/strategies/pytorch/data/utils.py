import numpy as np

from numba import njit

@njit(parallel=False)
def find_max_return(data, seq_len):
    data_len = data.shape[0] - seq_len
    max_val = np.zeros(data_len)

    print(data.shape)
    for i in range(data_len):
        # The last price is close price.
        close = data[i][-1]
        if not np.isnan(close):
            max_val[i] = np.fabs(data[i:i+seq_len, :] / close - 1).max()
    return max_val.max()
