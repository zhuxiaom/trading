# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:53:39 2019

@author: xzhu
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pyodbc as po
import pandas as pd
import json
from backtrader.feeds import PandasDirectData

class StockDB(object):
    def __init__(self):
        self.__conn = po.connect('DSN=stocks_db')

    def getQuotes(self, symbol, fromdate=None, todate=None, adjclose=True):
        cursor = self.__conn.cursor()
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        cursor.execute("""SELECT * FROM quotes WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = ?) ORDER BY q_date""", (symbol))
        for row in cursor.fetchall():
#            dt = pd.Timestamp(row[1].year, row[1].month, row[1].day)
            dates.append(pd.to_datetime(row[1]))
            ratio = 1.0
            if adjclose:
                ratio = row[6]*1.0/row[5]
#            dates.append(dt)
            opens.append(row[2]*ratio)
            highs.append(row[3]*ratio)
            lows.append(row[4]*ratio)
            closes.append(row[5]*ratio)
            volumes.append(row[7])

        quotes = pd.DataFrame(zip(dates, opens, highs, lows, closes, volumes))
        quotes = quotes.set_index(0)
        return PandasDirectData(dataname=quotes, fromdate=fromdate, todate=todate, openinterest=-1)

    def updateEtfDb(self, symbol, info, verbose=True):
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = ?""", (symbol))
        sym_id = cursor.fetchval()
        if sym_id:
            cursor.execute("""UPDATE symbols SET company_name = ? WHERE sym_id = ?""", (info['name'], sym_id))
            cursor.execute("""SELECT COUNT(*) FROM etf_info WHERE sym_id = ?""", (sym_id))
            if not cursor.fetchval():
                if verbose:
                    print('Insert ETF information of %s.' % (symbol))
                cursor.execute("""INSERT INTO etf_info VALUES(?, ?)""", (sym_id, json.dumps(info)))
            else:
                if verbose:
                    print('Update ETF information of %s.' % (symbol))
                cursor.execute("""UPDATE etf_info SET etfdb_info = ? WHERE sym_id = ?""", (json.dumps(info), sym_id))
            cursor.commit()
        else:
            print('Invalid symbol %s.' % (symbol))
