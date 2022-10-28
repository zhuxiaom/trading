from concurrent.futures import thread
import pandas as pd
import numpy as np
import argparse
from torch import frac, int32
import torch
import torch.utils.data as torch_data
from multiprocessing import Pool

from collections import defaultdict
from zenzic.data.watchlist import SP500
from zenzic.strategies.pytorch.data.utils import load_hist_quotes
from zenzic.thirdparty.Autoformer.utils.timefeatures import time_features
from numba import njit
from tqdm import tqdm
from torch.utils.data import TensorDataset

class Dataset(torch_data.Dataset):
    def __init__(self, filename, type, seq_len, channel_first=False):
        self.__seq_len = seq_len
        self.__channel_first = channel_first
        self.__quotes = pd.read_pickle(filename)
        self.__min_idx = self.__quotes['Symbol'].reset_index().groupby('Symbol').min()
        self.__trades = self.__quotes[
            ~self.__quotes['Label'].isna()][['Symbol', 'Date', 'Label']].reset_index()
        self.__trades.rename(columns={'index': 'Quotes Idx'}, inplace=True)
        self.__trades = self.__trades.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        self.__date_idx = self.__trades[['Date', 'Label']].groupby(by=['Date']).count()
        self.__date_idx = self.__date_idx.rename(columns={'Label': 'Count'}).reset_index(drop=False)
        self.__date_idx['Total'] = self.__date_idx['Count'].cumsum()
        self.__total_samples = self.__date_idx['Total'].iloc[-1]
        beg_idx = 0
        end_idx = np.searchsorted(self.__date_idx['Total'], self.__total_samples * 0.7, side='right')
        if type == 'val':
            beg_idx = np.searchsorted(self.__date_idx['Total'], self.__total_samples * 0.7, side='right')
            end_idx = np.searchsorted(self.__date_idx['Total'], self.__total_samples * 0.8, side='right')
        elif type == 'test':
            beg_idx = np.searchsorted(self.__date_idx['Total'], self.__total_samples * 0.8, side='right')
            end_idx = len(self.__date_idx)
        self.__date_idx = self.__date_idx.iloc[beg_idx:end_idx]
        self.__date_idx['Total'] = self.__date_idx['Count'].cumsum()
        self.__total_samples = self.__date_idx['Total'].iloc[-1]
        min_date = self.__date_idx['Date'].iloc[0]
        max_date = self.__date_idx['Date'].iloc[-1]
        self.__trades = self.__trades[
            (self.__trades['Date'] <= max_date) & (self.__trades['Date'] >= min_date)]
        # if type == 'train':
        #     # Shuffle the samples so that they are not sorted by (Symbol, Date).
        #     self.__trades = self.__trades.sample(frac=1).reset_index(drop=True)
        self.__cache = [None] * self.__total_samples

    def __len__(self):
        return self.__total_samples

    def __getitem__(self, index):
        if self.__cache[index] is None:
            # sym = self.__trades['Symbol'].iloc[index]
            # dt = self.__trades['Date'].iloc[index]
            # idx = self.__quotes.index[
            #     (self.__quotes['Symbol'] == sym) & (self.__quotes['Date'] == dt)].values[0]
            idx = self.__trades['Quotes Idx'].iloc[index]
            sym = self.__trades['Symbol'].iloc[index]
            min_idx = self.__min_idx.loc[sym]['index']
            beg_idx = idx - self.__seq_len
            padding = 0
            if beg_idx < min_idx:
                padding = self.__seq_len - (idx - min_idx)
                beg_idx = min_idx
            x = self.__quotes[['Open', 'High', 'Low', 'Close']].iloc[beg_idx:idx].values
            x = np.log(x / x[-1][-1])
            y = self.__quotes['Label'].iloc[idx]
            date_enc_x = np.vstack(self.__quotes['Date Enc'].iloc[beg_idx:idx].values)
            date_enc_y = self.__quotes['Date Enc'].iloc[idx]
            if padding != 0:
                x = np.vstack((np.zeros((padding, x.shape[-1])), x))
                date_enc_x = np.vstack((np.zeros((padding, date_enc_x.shape[-1])), date_enc_x))
            if self.__channel_first:
                x = np.transpose(x)
                date_enc_x = np.transpose(date_enc_x)
            self.__cache[index] =(
                x.astype(np.float32), date_enc_x.astype(np.float32),
                y.astype(np.int32), date_enc_y.astype(np.float32))
        x, date_enc_x, y, date_enc_y = self.__cache[index]
        return x, date_enc_x, y, date_enc_y

    def fetch(self, beg_idx, end_idx):
        return [self.__getitem__(i) for i in range(beg_idx, end_idx)]

    def to_gpu(self, device_name: str='cuda:0'):
        NUM_OF_THREADS = 4
        dst = torch.device(device=device_name)
        # self.__total_samples = 4
        step = self.__total_samples // NUM_OF_THREADS
        final_res = []
        with Pool(processes=NUM_OF_THREADS) as pool:
            param_lst = []
            for i in range(NUM_OF_THREADS):
                beg = i * step
                end = (i + 1) * step
                if i == (NUM_OF_THREADS - 1):
                    end = self.__total_samples
                param_lst.append((beg, end))
            for res in pool.starmap(self.fetch, param_lst):
                final_res.extend(res)
        x, date_x, y, date_y = tuple(zip(*final_res))
        return TensorDataset(
            torch.cuda.FloatTensor(np.asanyarray(x), device=dst),
            torch.cuda.IntTensor(np.asanyarray(date_x), device=dst),
            torch.cuda.FloatTensor(np.asanyarray(y), device=dst),
            torch.cuda.IntTensor(np.asanyarray(date_y), device=dst))
        
# Load WealthLab trades
def load_wl_trades(fname):
    int_t = lambda x:  np.NaN if x == 'Open' else int(x.replace(',', ''))
    float_t = lambda x: np.NaN if x == 'Open' else float(x.replace(',', ''))
    converters = {
        'Position': str,
        'Symbol': str,
        'Shares': int_t,
        'Entry Date': str,
        'Entry Price': float_t,
        'Exit Date': str,
        'Exit Price': float_t,
        '% Change': float_t,
        'Net Profit': float_t,
        'Bars Held': int_t,
        'Profit/Bar': float_t,
        'Entry Signal': str,
        'Exit Signal': str,
        'Cum Profit': float_t,
        'MAE %': float_t,
        'MFE %': float_t,
        'ChartScript': str,
    }
    trades = pd.read_csv(filepath_or_buffer=fname, sep='\t', converters=converters)
    trades = trades[~np.isnan(trades['Exit Price'])][['Symbol', 'Entry Date', 'Net Profit']]
    trades = trades.rename(columns={'Entry Date': 'Date'})
    trades['Date'] = pd.to_datetime(trades['Date'])
    trades['Label'] = trades['Net Profit'].apply(lambda x: 1 if x > 0.0 else 0)
    sp500 = SP500()
    trades = trades[trades['Symbol'].isin(sp500.get_symbols())]
    return trades

def load_quotes(symbols):
    quotes = load_hist_quotes(symbols)
    dates = pd.DataFrame(data=None, index=quotes['Date'].unique())
    date_enc =  time_features(dates.index, freq='d').transpose()
    dates['Date Enc'] = [x.astype('float32') for x in date_enc]
    quotes = quotes.join(dates, on=['Date'])
    return quotes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data samples from WealthLab trades.')
    parser.add_argument('--trades_file', type=str, required=True, default=None, help='The file of WealthLab trades.')
    parser.add_argument('--output', type=str, required=True, default=None, help='The output path of parquet file.')
    args = parser.parse_args()

    trades = load_wl_trades(args.trades_file)
    quotes = load_quotes(trades.Symbol.unique())
    trades = trades.set_index(keys=['Symbol', 'Date'])
    quotes = quotes.join(trades, on=['Symbol', 'Date'])
    quotes.to_pickle(args.output)
