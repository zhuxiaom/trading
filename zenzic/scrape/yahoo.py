import yfinance
import pandas
import numpy

from absl import flags
from absl import app
from absl import logging
from ratelimiter import RateLimiter
from timeit import Timer
from zenzic.data.stockdb import StockDB

@RateLimiter(max_calls=1, period=1)
def fetchHistQuote(symbol, period='1mo', proxy=None):
    yhoo = yfinance.Ticker(symbol)
    quote = yhoo.history(period, auto_adjust=False, rounding=False, proxy=proxy)
    try:
        quote.index = quote.index.tz_localize(None)
        quote = quote[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except:
        pass
    return quote
    
def syncQuotes(symbols, period="1mo", database=StockDB(), proxy=None):
    max_lst = []
    
    for sym in symbols:
        try:
            yahoo = fetchHistQuote(sym, period=period, proxy=proxy)
            yahoo = yahoo[~yahoo.index.duplicated(keep='first')]
        except Exception as ex:
            print("%s: error in fetching quote from Yahoo!" % (sym))
            print("Unexpected error: [%s] %s" % (type(ex).__name__, ex))
            continue
        if yahoo.empty:
            print("%s: Yahoo doesn't return any data." % (sym))
            continue
        
        db = database.getQuotes(sym, yahoo.index.min(), yahoo.index.max() + pandas.to_timedelta(1, unit='D'), adjclose=False, returnfmt="pandas")
        if period != "max" and db.empty:
            max_lst.append(sym)
            continue

        yahoo_only = yahoo.index.difference(db.index)
        db_only = db.index.difference(yahoo.index)
        common = yahoo.index.intersection(db.index)
        diff = common
        try:
            diff = common[~numpy.bitwise_and.reduce(numpy.isclose(yahoo.loc[common], db.loc[common]), axis=1)]
        except ValueError as e:
            # Assume all data with common index are different.
            print("%s: %s." % (sym, e))
        except TypeError as e:
            # Expected when either yahoo or db doesn't return any data.
            if len(common) != 0:
                raise
        precision = 2
        precision = yahoo[["Open", "High", "Low", "Close", "Adj Close"]].applymap(lambda x: ("%.6f" % (x)).rstrip("0")[::-1].find(".")).max().max()
        db = db.round(decimals=precision)

        print("%s: yahoo_only = %d, db_only = %d, common = %d, diff = %d, precision = %d, from = %s, to = %s." %
            (sym, yahoo_only.shape[0], db_only.shape[0], common.shape[0], diff.shape[0], precision, yahoo.index.min().strftime("%Y-%m-%d"), yahoo.index.max().strftime("%Y-%m-%d")))

        if period == "max":
            print("%s: resetting quotes." % (sym))
            database.updateQuotes(sym, yahoo, True)
        else:
            if db_only.shape[0] > 5:
                # Yahoo returns less quotes. We may want to retry.
                print("%s: Yahoo fails to return quotes on %s." % (sym, ", ".join([d.strftime("%Y-%m-%d") for d in db_only])))
                max_lst.append(sym)
                continue
            
            if (diff.shape[0] + yahoo_only.shape[0]) > float(yahoo.shape[0]) / 2:
                # More than half of Yahoo quotes are different. Reset the quotes.
                print("%s: too much difference and reset." % (sym))
                max_lst.append(sym)
                continue

            db_updates = pandas.concat([yahoo.loc[yahoo_only], yahoo.loc[diff]])
            database.updateQuotes(sym, db_updates)            
    
    if len(max_lst) > 0:
        syncQuotes(max_lst, period="max", database=database, proxy=proxy)

FLAGS = flags.FLAGS

flags.DEFINE_string("proxy", None, "Proxy setting.")
flags.DEFINE_integer("shard", -1, "The shard number.")
flags.DEFINE_string("period", "3mo", "The period to fetch qutoe from Yahoo.")
flags.DEFINE_string("start", None, "The first symbol in the shard to start with.")

def main(argv):
    stock_db = StockDB()
    symbols = stock_db.getStockSymbols(type="yahoo")
    # symbols = ["CLB",]
    # FLAGS.period = "max"
    half = len(symbols) // 2
    try:
        start = symbols.index(FLAGS.start) if FLAGS.start else 0
    except:
        start = 0
    if FLAGS.shard == 0:
        symbols = symbols[start:half] if start < half else symbols[0:half]
        start = 0
    elif FLAGS.shard == 1:
        symbols = symbols[start:] if start >= half else symbols[half:]
        start = 0
    print("Retrieve historical quotes for %d symbols." % (len(symbols)))
    syncQuotes(symbols[start:], period=FLAGS.period, database=stock_db, proxy={"https": FLAGS.proxy} if FLAGS.proxy else None)

if __name__ == "__main__":
    app.run(main)
    
