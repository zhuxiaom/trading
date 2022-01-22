import imp
import os
import pandas as pd
import numpy as np
import torch.utils.data as torch_data

from zenzic.thirdparty.Autoformer.utils.timefeatures import time_features
from zenzic.strategies.pytorch.data.utils import find_max_return

'''
    Dataset of Oxford Volatility downloaded from:
        https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
'''
class Dataset(torch_data.Dataset):
    # Symbols from the data.
    symbols = None
    # Volatility data cache avaiable at Class level.
    volatility = None
    # Scale factor at class level to be re-used.
    scale_factor = 1.0

    def __init__(self, data_file, flag='train',  size=None):
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

        self.__file_path = data_file
        self.__flag = flag

        if Dataset.symbols is None:
            Dataset.volatility = pd.DataFrame()
            self.__read_data()
            print('Finish reading data.')
            # Dataset.scale_factor = self.__get_scale_factor()
            # print("Scale factor is set to {}".format(Dataset.scale_factor))
        self.__volatility = Dataset.volatility.copy()

        # Trim dataset to 'train' (70%), 'val' (10%) or 'test' (20%)
        total_samples = self.__volatility.iloc[-1]['cum_samples']
        self.__beg_idx = 0
        self.__end_idx = np.searchsorted(
            self.__volatility['cum_samples'], total_samples * 0.7,  side='right')
        if flag == 'val':
            self.__beg_idx = np.searchsorted(
                self.__volatility['cum_samples'], total_samples * 0.7,  side='right')
            self.__end_idx = np.searchsorted(
                self.__volatility['cum_samples'], total_samples * 0.8,  side='right')
        elif flag == 'test':
            self.__beg_idx = np.searchsorted(
                self.__volatility['cum_samples'], total_samples * 0.8,  side='right')
            self.__end_idx = len(self.__volatility)
        print('Beg Idx: {}, End Idx: {}'.format(self.__beg_idx, self.__end_idx))
        # Make sure the order of symbol list is same as the order in 'row_index'.
        self.__symbols = [
            c.split(':')[0] for c in self.__volatility.filter(regex=(":close$")).columns]
        self.__volatility = self.__volatility.iloc[self.__beg_idx:]
        self.__end_idx = self.__end_idx - self.__beg_idx
        self.__beg_idx = 0
        self.__volatility['cum_samples'] = self.__volatility['num_samples'].cumsum()
        self.__total_samples = \
            self.__volatility.iloc[self.__end_idx - 1]['cum_samples']
        print('Total num of samples in {} dataset: {:,d}'.format(flag, self.__total_samples))

    def __read_data(self):
        data = pd.read_csv(self.__file_path)
        # data['date'] = data['Unnamed: 0'].apply(lambda x: pd.to_datetime(x, utc=True))
        # data.drop('Unnamed: 0', axis=1, inplace=True)
        data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        data.set_index(['Symbol', 'date'], inplace=True)
        # Assert no N/A values.
        assert(not data[['open_price', 'close_price', 'rv5_ss']].isna().any(axis=None)), 'NAN is found!'

        Dataset.symbols = data.index.unique(level=0)
        for s in Dataset.symbols:
            tmp = data.loc[s][['open_price', 'close_price', 'rv5_ss']]
            tmp.rename(columns={
                'open_price': s + ':open',
                'close_price': s + ':close',
                'rv5_ss': s + ':volatility'}, inplace=True)
            Dataset.volatility = Dataset.volatility.join(tmp, how='outer')
        Dataset.volatility.sort_index(inplace=True)
        for s in Dataset.symbols:
            index_col = s + ':index'
            Dataset.volatility[index_col] = Dataset.volatility[s + ':volatility'].apply(pd.notna).cumsum()
            Dataset.volatility[s + ':valid'] = self.__valid_sample(Dataset.volatility[index_col])
        date_enc =  time_features(
                pd.to_datetime(Dataset.volatility.index.values, utc=True), freq='h').transpose()
        Dataset.volatility['date_enc'] = [x.astype('float32') for x in date_enc]
        # Num of valid data samples each row.
        Dataset.volatility['num_samples'] = Dataset.volatility.filter(
            regex=(":valid$")).apply(lambda x: x.sum(), axis=1)
        # Cumulative sum of valid data samples
        Dataset.volatility['cum_samples'] = Dataset.volatility['num_samples'].cumsum()
        Dataset.volatility['row_index'] = Dataset.volatility.filter(regex=(":valid$")).apply(
            lambda x: np.array(x.cumsum()), axis=1)

    def __valid_sample(self, indices):
        max_index = indices[-1]
        valid = np.zeros(indices.shape, dtype=int)
        for i in range(len(indices)):
            last_index = indices[i] + self.__seq_len + self.__pred_len - 1
            if i == 0:
                if indices[i] > 0 and last_index <= max_index:
                    valid[i] = 1
            elif indices[i] > indices[i - 1]:
                if last_index <= max_index:
                    valid[i] = 1
        return valid

    def __get_scale_factor(self):
        max_val = np.zeros(len(Dataset.symbols))
        for i, sym in enumerate(Dataset.symbols):
            col_names = [sym + name for name in [':open', ':close']]
            data = Dataset.volatility.loc[:, col_names].dropna().values
            max_val[i] = find_max_return(data, self.__seq_len + self.__pred_len)
            print("{} max return: {}".format(sym, max_val[i]))
        return 1 / max_val.max()

    def __getitem__(self, index):
        row = np.searchsorted(
            self.__volatility['cum_samples'], index, side='right')
        index_base = (
            self.__volatility.iloc[row - 1]['cum_samples'] if row > 0 else 0)
        col = np.searchsorted(
            self.__volatility.iloc[row]['row_index'], index - index_base + 1, side='left')
        assert(col < len(self.__symbols)), 'col is {}, index is {}, len is {}, row is {}'.format(col, index, len(self.__symbols), row)
        sym = self.__symbols[col]
        col_names = [sym + name for name in [':open', ':close', ':volatility']] + ['date_enc']
        indices = self.__volatility[sym + ':index']
        last_index = np.searchsorted(indices, indices[row] + self.__seq_len + self.__pred_len, side='left')
        data = self.__volatility.iloc[row:last_index][col_names].copy()
        data.dropna(inplace=True)
        first_close = data.iloc[0][col_names[1]]
        data[col_names[:2]] = (data[col_names[:2]] / first_close - 1) * Dataset.scale_factor
        # print('row: {}, col: {}, sym: {}, first_close: {}'.format(row, col, sym, first_close))
        seq_x = data.iloc[:self.__seq_len][col_names[:3]].values
        seq_y = data.iloc[-(self.__label_len + self.__pred_len):][col_names[:3]].values
        seq_x_mark = np.vstack(data.iloc[:self.__seq_len][col_names[3]].values)
        seq_y_mark = np.vstack(data.iloc[-(self.__label_len + self.__pred_len):][col_names[3]].values)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.__total_samples

    def get_all_volatility(self):
        return self.__volatility

    def inverse_transform(self, v):
        return v / Dataset.scale_factor + 1


