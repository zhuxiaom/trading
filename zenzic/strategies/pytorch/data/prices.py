from numpy.ma.core import fabs
from pymysql.err import NotSupportedError
# from zenzic.thirdparty.Autoformer.utils.timefeatures import time_features
from zenzic.data.watchlist import SP500
from collections import defaultdict
from tqdm import tqdm, trange
from multiprocessing import Pool
import torch.utils.data as torch_data
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import argparse
import os
import pickle
import random

class Dataset(torch_data.Dataset):
    # Quote data cache avaiable at Class level.
    quotes_cache = defaultdict(lambda: pd.DataFrame())

    def __init__(self, watchlist='SP500', flag='train', seq_len=256, pred_len=24,
                 features='M', target='close', startdate=None):
        assert flag in ['train', 'test', 'val']

        self.__seq_len = seq_len
        self.__pred_len = pred_len

        # Those two flags are not used currently.
        self.__features = features
        self.__target = target

        db_conn = create_engine(
                'mysql+pymysql://localhost', connect_args={'read_default_file': '~/my.cnf'})
        self.__all_quotes = self.__get_all_quotes(watchlist, startdate, db_conn)
        if Dataset.quotes_cache['^GSPC'].empty:
            Dataset.quotes_cache['^GSPC'] = self.__get_quotes('^GSPC', startdate, db_conn)
        self.__sp500_quotes = Dataset.quotes_cache['^GSPC']

        # Trim dataset to 'train' (70%), 'val' (10%) or 'test' (20%)
        total_samples = \
            self.__all_quotes.iloc[-(self.__seq_len + self.__pred_len)]['cum_samples']
        beg_idx = 0
        end_idx = np.searchsorted(
            self.__all_quotes['cum_samples'], total_samples * 0.7,  side='right')
        if flag == 'val':
            beg_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.7,  side='right')
            end_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.85,  side='right')
        elif flag == 'test':
            beg_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.85,  side='right')
            end_idx = len(self.__all_quotes) - (self.__seq_len + self.__pred_len)
        print('Beg Idx: {}, End Idx: {}'.format(beg_idx, end_idx))
        self.__all_quotes = self.__all_quotes.iloc[
            beg_idx:(end_idx + self.__seq_len + self.__pred_len)].copy()
        # Make sure the order of symbol list is same as the order in 'row_index'.
        self.__symbols = [
            c.split(':')[0] for c in self.__all_quotes.filter(regex=(":close$")).columns]
        # Re-calcualte cumulative total of samples.
        self.__all_quotes['cum_samples'] = self.__all_quotes['num_samples'].cumsum()
        self.__total_samples = \
            self.__all_quotes.iloc[-(self.__seq_len + self.__pred_len)]['cum_samples']
        print('Total num of samples in {} dataset: {:,d}'.format(flag, self.__total_samples))

    def __get_all_quotes(self, watchlist, startdate, db_conn):
        if len(Dataset.quotes_cache[watchlist]) == 0:
            symbols = self.__get_symbols(watchlist)
            ''' The dataframe of quotes will be like:
                        A:open A:high A:low A:close ... num_samples cum_samples date_enc
            2020-01-01  1.0    2.0    3.0   4.0         1           1             (...)
            2020-01-02  2.0    3.0    4.0   5.0         1           2             (...)      
            '''
            all_quotes = pd.DataFrame()
            for s in tqdm(symbols, desc="Fetch stock prices"):
                all_quotes = all_quotes.join(self.__get_quotes(s, startdate, db_conn), how='outer')
            all_quotes.sort_index(inplace=True)
            # date_enc =  time_features(
            #     pd.to_datetime(all_quotes.index.values), freq='d').transpose()
            # all_quotes['date_enc'] = [x.astype('float32') for x in date_enc]
            all_quotes['date_enc'] = all_quotes.index.map(lambda x: x.year*10**4 + x.month*10**2 + x.day)
            # Num of valid data samples each row.
            all_quotes['num_samples'] = all_quotes.filter(
                regex=(":close$")).apply(lambda x: x.notna().sum(), axis=1)
            # Cumulative sum of valid data samples
            all_quotes['cum_samples'] = all_quotes['num_samples'].cumsum()
            all_quotes['row_index'] = all_quotes.filter(regex=(":close$")).apply(
                lambda x: np.array(x.notna().cumsum()), axis=1)
            # Copy to de-fragment data and improve performance.
            Dataset.quotes_cache[watchlist] = all_quotes.copy()

        return Dataset.quotes_cache[watchlist]

    def __adjust_price(self, x):
        # Replace NAN with close price then adjust to adj_close.
        x.fillna(x.iloc[3])
        return [(x.iloc[i]*x.iloc[4]/x.iloc[3]).astype('float32') for i in range(4)]

    def __get_quotes(self, symbol, startdate, db_conn):
        startdate = (startdate if startdate else '1900-01-01')
        query = """
            SELECT q_date, open, high, low, close, adj_close
            FROM quotes INNER JOIN symbols USING(sym_id)
            WHERE symbol = '{}' AND q_date >= '{}' ORDER BY q_date
        """.format(symbol, startdate)
        quotes = pd.read_sql(query, db_conn, index_col='q_date')
        # Adjust to adj_close
        quotes[['open', 'high', 'low', 'close']] = quotes.apply(
            lambda x: self.__adjust_price(x),
            axis=1, result_type='expand')
        quotes.drop(['adj_close'], axis=1, inplace=True)
        quotes.rename(
            columns=dict(zip(quotes.columns, [symbol+':'+c for c in quotes.columns])), inplace=True)
        invalid = quotes[quotes.values < 1e-6]
        if not invalid.empty:
            print(f"Found invalid prices for '{symbol}':")
            print(invalid)
            assert(False)
        return quotes

    def __get_symbols(self, watchlist):
        if watchlist == 'SP500':
            sp500 = SP500()
            return sp500.get_symbols()
        else:
            raise NotSupportedError("Watchlist {} is not supported!".format(watchlist))

    def __getitem__(self, index):
        row = np.searchsorted(
            self.__all_quotes['cum_samples'], index, side='right')
        index_base = (
            self.__all_quotes.iloc[row - 1]['cum_samples'] if row > 0 else 0)
        col = np.searchsorted(
            self.__all_quotes.iloc[row]['row_index'], index - index_base + 1, side='left')
        sym = self.__symbols[col]
        price_cols = [sym + name for name in [':open', ':high', ':low', ':close']]
        date_enc_col = 'date_enc'
        data = self.__all_quotes.iloc[row:(row + self.__seq_len + self.__pred_len)][price_cols + [date_enc_col]]
        num_of_rows = data.shape[0]
        data = data.join(self.__sp500_quotes, how='inner').copy()
        price_cols.extend(['^GSPC:open', '^GSPC:high', '^GSPC:low', '^GSPC:close'])
        # print("Price colums: {}".format(price_cols))
        assert(num_of_rows == data.shape[0]), "Num of rows doesn't match! {} v.s. {} on index {}".format(num_of_rows, data.shape[0], index)
        seq_x = data.iloc[:self.__seq_len][price_cols].values
        seq_y = data.iloc[-self.__pred_len:][price_cols].values
        # convert to symbol returns.
        close = seq_x[-1][3]
        seq_x[:, :4] = close / seq_x[:, :4]
        seq_y[:, :4] = seq_y[:, :4] / close
        assert seq_x[-1][3] == 1.0, "Failed to convert symbols return!"
        # convert to sp500 returns.
        close = seq_x[-1][-1]
        seq_x[:, -4:] = close / seq_x[:, -4:]
        seq_y[:, -4:] = seq_y[:, -4:] / close
        assert seq_x[-1][-1] == 1.0, "Failed to convert SP500 return!"
        seq_x_mark = np.vstack(data.iloc[:self.__seq_len][date_enc_col].values)
        seq_y_mark = np.vstack(data.iloc[-self.__pred_len:][date_enc_col].values)

        return seq_x, seq_x_mark, seq_y, seq_y_mark

    def __len__(self):
        return self.__total_samples

    def fetch(self, beg_idx, end_idx):
        return [self.__getitem__(i) for i in range(beg_idx, end_idx)]
    
    def fetch_all(self, offset: int=0):
        NUM_OF_THREADS = 8
        # self.__total_samples = 640
        final_res = []
        range_lb = (-offset if offset < 0 else 0)
        range_ub = (offset if offset > 0 else self.__total_samples)
        total = range_ub - range_lb
        step = total // NUM_OF_THREADS
        print(f"Fetching samples from {range_lb} to {range_ub}.")
        beg = range_lb
        with Pool(processes=NUM_OF_THREADS) as pool:
            param_lst = []
            for i in range(NUM_OF_THREADS):
                end = beg + step
                if i == (NUM_OF_THREADS - 1):
                    end = range_ub
                param_lst.append((beg, end))
                beg = end
            for res in pool.starmap(self.fetch, param_lst):
                final_res.extend(res)
        x, date_x, y, date_y = tuple(zip(*final_res))
        return x, date_x, y, date_y

def save_dataset(args, type):
    # x, date_x, y, date_y = Dataset(
    #     flag=type, seq_len=args.seq_len, pred_len=args.pred_len, startdate=args.start_date).fetch_all()
    # data = dict(
    #     x = x,
    #     date_x = date_x,
    #     y = y,
    #     date_y = date_y
    # )
    dataset = Dataset(
        flag=type, seq_len=args.seq_len, pred_len=args.pred_len, startdate=args.start_date)
    select_k = round(args.downsample * len(dataset))
    x = np.zeros((select_k, args.seq_len, dataset[0][0].shape[1]), dtype=np.float32)
    date_x = np.zeros((select_k, args.seq_len, dataset[0][1].shape[1]), dtype=np.float32)
    y = np.zeros((select_k, args.pred_len, dataset[0][2].shape[1]), dtype=np.float32)
    date_y = np.zeros((select_k, args.pred_len, dataset[0][3].shape[1]), dtype=np.float32)
    for i in trange(len(dataset), desc="Fetch {} samples: ".format(type)):
        if i < select_k:
            x[i], date_x[i], y[i], date_y[i] = dataset[i]
        else:
            m = random.randint(0, i)
            if m < select_k:
                x[m], date_x[m], y[m], date_y[m] = dataset[i]
    print("Collected {} samples in {} dataset.".format(x.shape[0], type))
    data = dict(
        x = x,
        date_x = date_x,
        y = y,
        date_y = date_y
    )
    path = os.path.join(args.output_dir, type + '.pkl')
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate prices datasets.')
    parser.add_argument('--seq_len', type=int, default=256, help='The sequence length.')
    parser.add_argument('--pred_len', type=int, default=24, help='The prediction length.')
    parser.add_argument('--start_date', type=str, default='2000-01-01', help='The starting date of stock price.')
    parser.add_argument('--downsample', type=float, default=1.0, help='The ratio of downsampling.')
    parser.add_argument('--output_dir', type=str, required=True, default=None, help='The output directory.')
    args = parser.parse_args()

    save_dataset(args, 'train')
    save_dataset(args, 'val')
    save_dataset(args, 'test')

