from email.quoprimime import quote
from matplotlib.pyplot import axis
from zenzic.data.watchlist import SP500
from zenzic.data.stockdb import StockDB
from tqdm import tqdm

import pandas as pd
import numpy as np
import math

def sanity_check(date_idx, quotes):
    passed = True
    for sym in tqdm(quotes['Symbol'].unique(), desc='Data sanity check'):
        q = quotes[quotes['Symbol'] == sym]
        min_date = q['Date'].min()
        q = q.join(date_idx[date_idx.index >= min_date], on='Date', how='right')
        missing = q[q['Symbol'].isna()]['Date'].to_list()
        if len(missing) > 0:
            print(f"{sym}: missing dates {missing}.")
            passed = False
    return passed

def sp500_prices(start_date=None, end_date=None):
    symbols = SP500().get_symbols()
    db = StockDB()
    quotes = []
    for s in tqdm(symbols, desc='Reading price data'):
        q = db.getQuotes(symbol=s, fromdate=start_date, todate=end_date, returnfmt='pandas')
        q['Symbol'] = s
        q.reset_index(inplace=True)
        # q = q.set_index(['Symbol', 'Date'])
        # Calculate log returns and drop the first row.
        assert(q.notna().all(axis=None)), "Found N/A values of symbol '{}'.".format(s)
        q['Prev_close'] = q['Close'].shift(1).bfill()
        q['Open'] = np.log(q['Open'] / q['Prev_close'])
        q['High'] = np.log(q['High'] / q['Prev_close'])
        q['Low'] = np.log(q['Low'] / q['Prev_close'])
        q['Close'] = np.log(q['Close'] / q['Prev_close'])
        q.drop(columns=['Adj Close', 'Volume', 'Prev_close'], inplace=True)
        quotes.append(q)
    quotes = pd.concat(quotes)
    date_idx = pd.DataFrame()
    date_idx['Date'] = quotes['Date'].unique()
    date_idx = date_idx.set_index(['Date'])
    date_idx['Date_idx'] = [i for i in range(len(date_idx))]
    assert(sanity_check(date_idx, quotes)), "Failed sanity check!"
    quotes = quotes.join(date_idx, on='Date', how='left').set_index(['Symbol', 'Date'])
    quotes = quotes.reset_index()
    return quotes

def split_data(data, seq_len, pred_len, cutoff):
    sample_len = seq_len + pred_len
    max_date_idx = data.groupby('Symbol').max()['Date_idx'].max()
    assert((data.groupby('Symbol').max()['Date_idx'] == max_date_idx).all()), 'Max date_idx is not aligned!'
    symbols = data['Symbol'].unique()
    total_samples = (
        data.groupby('Symbol').max()['Date_idx'] - data.groupby('Symbol').min()['Date_idx'] + 1 - sample_len + 1).sum()
    pred_days = math.ceil(total_samples * (1 - cutoff) / len(symbols))
    cutoff_idx = max_date_idx + 1 - pred_days
    df_train = data[data['Date_idx'] < cutoff_idx]
    df_pred = data[data['Date_idx'] >= (cutoff_idx - seq_len)]

    pred_total_samples = df_pred.groupby('Symbol').max()['Date_idx'] - df_pred.groupby('Symbol').min()['Date_idx'] + 1
    if not (pred_total_samples >= sample_len).all():
        df_pred = df_pred[~df_pred['Symbol'].isin(pred_total_samples[pred_total_samples < sample_len].index)]
        # print("Remove symbols {} from pred dataset.".format(pred_total_samples[pred_total_samples < sample_len].index.tolist()))
    train_total_samples = df_train.groupby('Symbol').max()['Date_idx'] - df_train.groupby('Symbol').min()['Date_idx'] + 1
    if not (train_total_samples >= sample_len).all():
        df_train = df_train[~df_train['Symbol'].isin(train_total_samples[train_total_samples < sample_len].index.values)]
        # print("Remove symbols {} from train dataset.".format(train_total_samples[train_total_samples < sample_len].index.tolist()))
    return df_train, df_pred

def date_features(data):
    features = pd.DataFrame()
    features['Date'] = data['Date'].unique()
    features['Day_of_year'] = features['Date'].apply(lambda x: x.day_of_year)
    features['Day_of_week'] = features['Date'].apply(lambda x: x.day_of_week)
    features['Month'] = features['Date'].apply(lambda x: x.month)
    features['Day'] = features['Date'].apply(lambda x: x.day)
    features = features.set_index(['Date'])
    return features

if __name__ == '__main__':
    sp500_prices(start_date='1999-01-01')
    
