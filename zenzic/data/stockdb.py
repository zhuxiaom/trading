# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:53:39 2019

@author: xzhu
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pymysql as mysql
import pandas as pd
import json
from backtrader.feeds import PandasDirectData
from datetime import date
from sqlalchemy import create_engine

class StockDB(object):
    def __init__(self):
        self.__conn = mysql.connect(read_default_file="~/my.cnf")
        self.__sqlalchemy = create_engine('mysql+pymysql://localhost', connect_args={'read_default_file': '~/my.cnf'})
        self.__today = date.today()

    def getQuotes(self, symbol, fromdate=None, todate=None, adjclose=True, returnfmt=None):
        cursor = self.__conn.cursor()
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        adj_closes = []
        volumes = []
        query = """
            SELECT q_date, open, high, low, close, adj_close, volume
            FROM quotes
            WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = '{}')
        """.format(symbol)
        if fromdate:
            query += " AND q_date >= '{}'".format(pd.to_datetime(fromdate).date())
        if todate:
            query += " AND q_date < '{}'".format(pd.to_datetime(todate).date())
        quotes = pd.read_sql(query, self.__sqlalchemy)
        quotes['q_date'] = pd.to_datetime(quotes['q_date'])

        if adjclose:
            quotes['ratio'] = quotes['adj_close'] / quotes['close']
            quotes['open'] = quotes['open'] * quotes['ratio']
            quotes['high'] = quotes['high'] * quotes['ratio']
            quotes['low'] = quotes['low'] * quotes['ratio']
            quotes['close'] = quotes['close'] * quotes['ratio']
            quotes.drop(columns=['ratio'], inplace=True)
        quotes.rename(
            columns={'q_date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'adj_close': 'Adj Close', 'volume': 'Volume'},
            inplace=True)
        quotes.set_index('Date', inplace=True)
        if returnfmt == 'pandas':
            return quotes
        else:
            return PandasDirectData(dataname=quotes, fromdate=fromdate, todate=todate, openinterest=-1)

    def getStockSymbols(self, type="finviz"):
        symbols = []
        already_updated = {}
        cursor = self.__conn.cursor()
        if type == "finviz":
            cursor.execute("""SELECT s.symbol, MAX(i.u_date) FROM symbols s JOIN stock_info i USING(sym_id) GROUP BY symbol""")
            for row in cursor.fetchall():
                if row[1] == self.__today:
                    already_updated[row[0]] = row[1]
            cursor.execute("""SELECT s.symbol FROM symbols s LEFT JOIN etf_info e USING(sym_id) WHERE s.yhoo_sync AND e.u_date IS NULL AND s.symbol NOT LIKE '^%' AND exchange_id != 8 ORDER BY symbol""")
            for row in cursor.fetchall():
                if not already_updated.get(row[0], None):
                    symbols.append(row[0])
        elif type == "yahoo":
            cursor.execute("""SELECT symbol FROM symbols WHERE yhoo_sync ORDER BY symbol""")
            for row in cursor.fetchall():
                symbols.append(row[0])
        
        return symbols

    def updateEtfDb(self, symbol, info):
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = %s""", (symbol))
        sym_id = cursor.fetchone()
        if sym_id:
            # cursor.execute("""UPDATE symbols SET company_name = %s WHERE sym_id = %d""", (info['name'], sym_id))
            cursor.execute("""INSERT INTO etf_info VALUES(%s, %s, %s)""", (sym_id[0], self.__today, json.dumps(info)))
            self.__conn.commit()
        else:
            print('Invalid symbol %s.' % (symbol))

    def updateStockInfo(self, symbol, info):
        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = %s""", (symbol))
        sym_id = cursor.fetchone()[0]
        if sym_id:
            if len(info) > 0:
                cursor.execute("""INSERT INTO stock_info VALUES(%s, %s, %s)""", (sym_id, self.__today, json.dumps(info)))
            else:
                cursor.execute("""UPDATE symbols SET yhoo_sync = 0 WHERE symbol = %s""", (symbol))
            self.__conn.commit()
        else:
            print('Invalid symbol %s.' % (symbol))

    def __updateSingleRow(self, cursor, sym_id, row):
        try:
            sql = (
                "INSERT INTO quotes (sym_id, q_date, open, high, low, close, adj_close, volume) "
                "VALUES(%s, %s, %s, %s, %s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE open=new.open, high=new.high, low=new.low, close=new.close, adj_close=new.adj_close, volume=new.volume"
            )

            if row.shape[0] != 0 and not row.isnull().any():
                assert(row.shape[0] == 6)
                q_date = row.name.to_pydatetime()
                open_p = row["Open"].item()
                high_p = row["High"].item()
                low_p = row["Low"].item()
                close_p = row["Close"].item()
                adj_close_p = row["Adj Close"].item()
                volume = int(row["Volume"].item())
                cursor.execute(sql, (sym_id, q_date, open_p, high_p, low_p, close_p, adj_close_p, volume))
        except Exception as ex:
            print("Failed update row: ", row)
            print("Unexpected error: ", ex)
            raise

    def updateQuotes(self, symbol, quotes, reset=False):
        if quotes.shape[0] == 0:
            return

        cursor = self.__conn.cursor()
        cursor.execute("""SELECT sym_id FROM symbols WHERE symbol = %s""", (symbol))
        sym_id = cursor.fetchone()[0]
        if sym_id:
            try:
                if reset:
                    cursor.execute("DELETE FROM quotes WHERE sym_id = %s", (sym_id))
                quotes.apply(lambda row: self.__updateSingleRow(cursor, sym_id, row), axis=1)
                self.__conn.commit()
            except:
                self.__conn.rollback()
                pass
        else:
            print('Invalid symbol %s.' % (symbol))
