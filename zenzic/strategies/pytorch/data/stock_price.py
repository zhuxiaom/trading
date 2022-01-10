from numpy.ma.core import fabs
from pymysql.err import NotSupportedError
from zenzic.thirdparty.Autoformer.utils.timefeatures import time_features
from collections import defaultdict
from tqdm import tqdm, trange
from numba import njit
import torch.utils.data as torch_data
import pymysql as mysql
import pandas as pd
import numpy as np

@njit(parallel=True)
def find_max_return(data, seq_len):
    data_len = data.shape[0] - seq_len
    max_val = np.zeros(data_len)

    for i in range(data_len):
        close = data[i][3]
        if not np.isnan(close):
            max_val[i] = np.fabs(data[i:i+seq_len, :] / close - 1).max()
    return max_val.max()

class Dataset(torch_data.Dataset):
    # Quote data cache avaiable at Class level.
    quotes_cache = defaultdict(lambda: pd.DataFrame())
    # Scale factor at class level to be re-used.
    scale_factor = None

    def __init__(self, watchlist='SP500', flag='train', size=None,
                 features='M', target='close', startdate=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.__seq_len = 24 * 4 * 4
            self.__label_len = 24 * 4
            self.__pred_len = 24 * 4
        else:
            self.__seq_len = size[0]
            self.__label_len = size[1]
            self.__pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert self.__label_len <= self.__seq_len

        # Those two flags are not used currently.
        self.__features = features
        self.__target = target

        self.__db_conn = mysql.connect(read_default_file="~/my.cnf")
        self.__symbols = self.__get_symbols(watchlist).copy()
        self.__all_quotes = self.__get_all_quotes(watchlist, startdate)

        if Dataset.scale_factor is None:
            Dataset.scale_factor = self.__get_scale_factor(watchlist)
        print("Scale factor is set to {}".format(Dataset.scale_factor))
        
        # Trim dataset to 'train' (70%), 'val' (10%) or 'test' (20%)
        total_samples = \
            self.__all_quotes.iloc[-(self.__seq_len + self.__pred_len + 1)]['cum_samples']
        beg_idx = 0
        end_idx = np.searchsorted(
            self.__all_quotes['cum_samples'], total_samples * 0.7,  side='right')
        if flag == 'val':
            beg_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.7,  side='right')
            end_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.8,  side='right')
        elif flag == 'test':
            beg_idx = np.searchsorted(
                self.__all_quotes['cum_samples'], total_samples * 0.8,  side='right')
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
            self.__all_quotes.iloc[-(self.__seq_len + self.__pred_len + 1)]['cum_samples']
        print('Total num of samples in {} dataset: {:,d}'.format(flag, self.__total_samples))

    def __get_all_quotes(self, watchlist, startdate):
        if len(Dataset.quotes_cache[watchlist]) == 0:
            ''' The dataframe of quotes will be like:
                        A:open A:high A:low A:close ... num_samples cum_samples date_enc
            2020-01-01  1.0    2.0    3.0   4.0         1           1             (...)
            2020-01-02  2.0    3.0    4.0   5.0         1           2             (...)      
            '''
            all_quotes = pd.DataFrame()
            for s in tqdm(self.__symbols, desc="Fetching quotes"):
                all_quotes = all_quotes.join(self.__get_quotes(s, startdate), how='outer')
            all_quotes.sort_index(inplace=True)
            date_enc =  time_features(
                pd.to_datetime(all_quotes.index.values), freq='h').transpose()
            all_quotes['date_enc'] = [x.astype('float32') for x in date_enc]
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
        x.fillna(x[3])
        return [(x[i]*x[4]/x[3]).astype('float32') for i in range(4)]

    def __get_quotes(self, symbol, startdate):
        startdate = (startdate if startdate else '1900-01-01')
        query = """
            SELECT q_date, open, high, low, close, adj_close
            FROM quotes INNER JOIN symbols USING(sym_id)
            WHERE symbol = '{}' AND q_date >= '{}' ORDER BY q_date
        """.format(symbol, startdate)
        quotes = pd.read_sql(query, self.__db_conn, index_col='q_date')
        # Adjust to adj_close
        quotes[['open', 'high', 'low', 'close']] = quotes.apply(
            lambda x: self.__adjust_price(x),
            axis=1, result_type='expand')
        quotes.drop(['adj_close'], axis=1, inplace=True)
        quotes.rename(
            columns=dict(zip(quotes.columns, [symbol+':'+c for c in quotes.columns])), inplace=True)
        # print('Retrieved {} quotes for {}'.format(len(quotes), symbol))
        return quotes

    def __get_symbols(self, watchlist):
        if watchlist == 'SP500':
            query = """
                SELECT symbol AS symbol FROM symbols JOIN stock_info
                USING(sym_id) WHERE finviz_info->>'$.Index' LIKE '%S&P%'
                AND u_date = (SELECT MAX(u_date) FROM stock_info)
                ORDER BY symbol
            """
            symbols = pd.read_sql(query, self.__db_conn, index_col='symbol')
            try:
                # Remove duplicated assets.
                symbols.drop(['DISCK', 'FOX', 'GOOG', 'NWS', 'UA'], inplace=True)
            except:
                pass
            assert(len(symbols) == 498), 'Unexpected watchlist length {}!'.format(len(symbols))
            return symbols.index.to_list()
        else:
            raise NotSupportedError("Watchlist {} is not supported!".format(watchlist))

    def __get_scale_factor(self, watchlist):
        max_val = np.zeros(len(self.__symbols))
        for i in trange(len(self.__symbols), desc="Adjusting scale"):
            sym = self.__symbols[i]
            col_names = [sym + name for name in [':open', ':high', ':low', ':close']]
            data = Dataset.quotes_cache[watchlist].loc[:, col_names].values
            max_val[i] = find_max_return(data, self.__seq_len + self.__pred_len)
            # print("{} max return: {}".format(sym, max_val[i]))
        return 1 / max_val.max()

    def __getitem__(self, index):
        row = np.searchsorted(
            self.__all_quotes['cum_samples'], index, side='right')
        index_base = (
            self.__all_quotes.iloc[row - 1]['cum_samples'] if row > 0 else 0)
        col = np.searchsorted(
            self.__all_quotes.iloc[row]['row_index'], index - index_base + 1, side='left')
        sym = self.__symbols[col]
        col_names = [sym + name for name in [':open', ':high', ':low', ':close']] + ['date_enc']
        data = self.__all_quotes.iloc[row:(row + self.__seq_len + self.__pred_len)][col_names].copy()
        first_close = data.iloc[0][col_names[3]]
        # print('row: {}, col: {}, sym: {}, first_close: {}'.format(row, col, sym, first_close))
        seq_x = ((data.iloc[:self.__seq_len][col_names[:4]].values / first_close) - 1) * Dataset.scale_factor
        seq_y = ((data.iloc[-(self.__label_len + self.__pred_len):][col_names[:4]].values / first_close) - 1) * Dataset.scale_factor
        seq_x_mark = np.vstack(data.iloc[:self.__seq_len][col_names[4]].values)
        seq_y_mark = np.vstack(data.iloc[-(self.__label_len + self.__pred_len):][col_names[4]].values)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.__total_samples

    def get_all_quotes(self):
        return self.__all_quotes

    def inverse_transform(self, v):
        return v / Dataset.scale_factor + 1
