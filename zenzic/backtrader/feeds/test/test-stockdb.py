# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:46:35 2019

@author: xzhu
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import backtrader as bt
from zenzic.backtrader.feeds import StockDB

def runstrat():
    args = parse_args()

    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add a strategy
    cerebro.addstrategy(bt.Strategy)

    # Pass it to the backtrader datafeed and add it to the cerebro
    db = StockDB()

    cerebro.adddata(db.getQuotes('GOOG'))

    # Run over everything
    cerebro.run()

    # Plot the result
    cerebro.plot(style='bar', iplot=False, use='Qt5Agg')
    print("Execution is complete!")


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
