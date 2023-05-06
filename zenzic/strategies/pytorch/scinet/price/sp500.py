import argparse
import datetime as dt
import pickle
import pandas as pd

from sqlalchemy import create_engine, text
from multiprocessing import Pool
from tqdm import tqdm, trange
from zenzic.data.watchlist import SP500
from zenzic.thirdparty.Autoformer.utils.timefeatures import time_features


class HistPrices:
    def __init__(self, args) -> None:
        sp500 = SP500()
        self.__symbols = sp500.get_symbols()
        # print(self.__symbols)
        self.__db_conn = None
        self.__s_date = args.start
        self.__e_date = args.end
        self.__seq_len = args.seq_len
        self.__pred_len = args.pred_len
        self.__sp500_quotes = None
        self.__n_threads = args.n_threads

    def __get_quotes(self, symbol: str, s_date: str, e_date: str):
        query = """
            SELECT q_date, open, high, low, close, adj_close
            FROM quotes INNER JOIN symbols USING(sym_id)
            WHERE symbol = '{}' AND q_date >= '{}'  AND q_date < '{}'
            ORDER BY q_date
        """.format(symbol, s_date, e_date)
        query = text(query)
        quotes = pd.read_sql(query, self.__db_conn, index_col='q_date')
        if quotes.shape[0] > 0:
            # Adjust to adj_close
            quotes[['open', 'high', 'low', 'close']] = quotes.apply(
                lambda x: self.__adjust_price(x),
                axis=1, result_type='expand')
        quotes.drop(['adj_close'], axis=1, inplace=True)
        quotes.rename(
            columns=dict(zip(quotes.columns, [symbol+':'+c for c in quotes.columns])), inplace=True)
        # print('Retrieved {} quotes for {}'.format(len(quotes), symbol))
        return quotes

    def __adjust_price(self, x):
        # Replace NAN with close price then adjust to adj_close.
        x.fillna(x[3])
        return [(x[i]*x[4]/x[3]).astype('float32') for i in range(4)]

    def fetch(self, sym: str):
        if not self.__db_conn:
            self.__db_conn = create_engine(
                'mysql+pymysql://127.0.0.1', connect_args={'read_default_file': '~/my.cnf'})
            self.__sp500_quotes = self.__get_quotes(
                '^GSPC', self.__s_date, self.__e_date)
            self.__sp500_quotes.drop(
                ['^GSPC:open', '^GSPC:high', '^GSPC:low'], axis=1, inplace=True)

        quotes = self.__get_quotes(sym, self.__s_date, self.__e_date)
        quotes = quotes.join(self.__sp500_quotes, how='outer')
        quotes.sort_index(inplace=True)
        nan_values = quotes[quotes.isna().any(axis=1)]
        if not nan_values.empty:
            quotes = quotes[quotes.index > nan_values.index.max()]
            print('Generate samples of {} since {} (shorter than requested)!'.format(
                sym, quotes.index.min()))
        else:
            print('Generate samples of {} since {}.'.format(
                sym, quotes.index.min()))
        np_quotes = quotes.to_numpy()
        samples = []
        oldest_sample_end = dt.date.today()
        for i in range(self.__seq_len, len(np_quotes)-self.__pred_len+1):
            oldest_sample_end = min(
                quotes.index[i+self.__pred_len-1], oldest_sample_end)
            samples.append(
                (np_quotes[i-self.__seq_len:i], np_quotes[i:i+self.__pred_len, 3]))
        return samples, oldest_sample_end

    def samples(self):
        results = []
        oldest_sample_end = dt.date.today()
        with Pool(processes=self.__n_threads) as pool:
            for res in pool.starmap(self.fetch, [*zip(self.__symbols)]):
                results.extend(res[0])
                oldest_sample_end = min(res[1], oldest_sample_end)
        print('The oldest sample ends at {}.'.format(oldest_sample_end))
        return results


def parse_date(d: str):
    return dt.datetime.strptime(d, r'%Y-%m-%d').date().strftime(r'%Y-%m-%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate data samples from WealthLab trades.')
    parser.add_argument('--start', type=parse_date, required=True,
                        default=None, help='The starting date (inclusive).')
    parser.add_argument('--end', type=parse_date, required=True,
                        default=None, help='The end date (exclusive).')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='The max length of the price series.')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='The length of predicted prices series.')
    parser.add_argument('--out_file', type=str, default=None,
                        help='The output pickle file.')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='The num of threads to process data.')
    args = parser.parse_args()
    samples = HistPrices(args).samples()
    print('Total num of samples: {}'.format(len(samples)))
    if args.out_file:
        print('Saving samples to: {}.'.format(args.out_file))
        with open(args.out_file, "wb") as f:
            pickle.dump(samples, f)
