from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import backtrader as bt
from backtrader.indicators import WeightedMovingAverage
from zenzic.data import StockDB
from zenzic.indicators import LgFilter
import matplotlib
import pandas as pd

def runstrat():
    args = parse_args()  # pylint: disable=unused-variable
    
    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)    # pylint: disable=unexpected-keyword-arg

    # Add a strategy
    cerebro.addstrategy(bt.Strategy)

    # Pass it to the backtrader datafeed and add it to the cerebro
    db = StockDB()
    goog = db.getQuotes('GOOG', fromdate=pd.to_datetime('2019-01-01'))
    cerebro.adddata(goog)
    cerebro.addindicator(LgFilter, data=cerebro.datas[0].close, gamma=0.4)
    # cerebro.addindicator(WeightedMovingAverage)

    # Run over everything
    cerebro.run()

    matplotlib.use('Qt5Agg')

    # Plot the result
    cerebro.plot(style='bar', iplot=False)

    print(matplotlib.get_backend())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()


if __name__ == '__main__':
    runstrat()
