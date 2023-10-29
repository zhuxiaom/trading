import numpy as np
import pandas as pd

from numba import njit
from tqdm import tqdm, trange
from zenzic.data.stockdb import StockDB

@njit(parallel=True)
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

def load_hist_quotes(symbols, fromdate=None, enddate=None):
    all_quotes = pd.DataFrame()
    quotes_lst = []
    stock_db = StockDB()
    for sym in tqdm(symbols, desc='Load historical quotes'):
        quotes = stock_db.getQuotes(
            sym, fromdate=fromdate, todate=enddate, returnfmt='pandas').reset_index()
        symbol = pd.DataFrame(data=[sym] * quotes.shape[0], columns=['Symbol'])
        quotes = pd.concat(objs=[symbol, quotes], axis=1)
        quotes_lst.append(quotes)
    all_quotes = pd.concat(quotes_lst, ignore_index=True)
    all_quotes = all_quotes.sort_values(by=['Symbol', 'Date'])
    all_quotes = all_quotes.reset_index(drop=True)
    return all_quotes