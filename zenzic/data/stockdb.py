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
from datetime import date

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

    def getStockSymbols(self):
        symbols = []
        already_updated = {}
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT s.symbol, MAX(i.u_date) FROM symbols s JOIN stock_info i USING(sym_id) GROUP BY symbol""")
        for row in cursor.fetchall():
            if row[1] == date.today():
                already_updated[row[0]] = row[1]
        cursor.execute("""SELECT s.symbol FROM symbols s LEFT JOIN etf_info e USING(sym_id) WHERE s.yhoo_sync AND e.u_date IS NULL AND s.symbol NOT LIKE '^%' ORDER BY symbol""")
        for row in cursor.fetchall():
            if not already_updated.get(row[0], None):
                symbols.append(row[0])
        return symbols

    def updateEtfDb(self, symbol, info):
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = ?""", (symbol))
        sym_id = cursor.fetchval()
        if sym_id:
            cursor.execute("""UPDATE symbols SET company_name = ? WHERE sym_id = ?""", (info['name'], sym_id))
            cursor.execute("""INSERT INTO etf_info VALUES(?, ?, ?)""", (sym_id, date.today(), json.dumps(info)))
            cursor.commit()
        else:
            print('Invalid symbol %s.' % (symbol))

    def updateStockInfo(self, symbol, info):
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = ?""", (symbol))
        sym_id = cursor.fetchval()
        if sym_id:
            cursor.execute("""INSERT INTO stock_info VALUES(?, ?, ?)""", (sym_id, date.today(), json.dumps(info)))
            cursor.commit()
        else:
            print('Invalid symbol %s.' % (symbol))
