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

class StockDB(object):
    def __init__(self):
        self.__conn = mysql.connect(read_default_file="~/my.cnf")
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
        s_fromdate = (fromdate.to_pydatetime() if fromdate else None)
        s_todate = (todate.to_pydatetime() if todate else None)
        if s_fromdate and s_todate:
            cursor.execute("""SELECT * FROM quotes WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = %s) AND q_date >= %s AND q_date <= %s ORDER BY q_date""", (symbol, s_fromdate, s_todate))
        elif s_fromdate:
            cursor.execute("""SELECT * FROM quotes WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = %s) AND q_date >= %s ORDER BY q_date""", (symbol, s_fromdate))
        elif s_todate:
            cursor.execute("""SELECT * FROM quotes WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = %s) AND q_date <= %s ORDER BY q_date""", (symbol, s_todate))
        else:
            cursor.execute("""SELECT * FROM quotes WHERE sym_id = (SELECT sym_id FROM symbols WHERE symbol = %s) ORDER BY q_date""", (symbol))
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
            adj_closes.append(row[6])
            volumes.append(row[7])
        
        if returnfmt == 'pandas':
            quotes = pd.DataFrame({"Date": dates, "Open": opens, "High": highs, "Low": lows, "Close": closes, "Adj Close": adj_closes, "Volume": volumes})
            quotes = quotes.set_index("Date")
            return quotes
        else:
            quotes = pd.DataFrame({"Date": dates, "Open": opens, "High": highs, "Low": lows, "Close": closes, "Adj Close": adj_closes, "Volume": volumes})
            quotes = quotes.set_index("Date")
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
            cursor.execute("""SELECT s.symbol FROM symbols s LEFT JOIN etf_info e USING(sym_id) WHERE s.yhoo_sync AND e.u_date IS NULL AND s.symbol NOT LIKE '^%' ORDER BY symbol""")
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
        sym_id = cursor.fetchone()[0]
        if sym_id:
            # cursor.execute("""UPDATE symbols SET company_name = %s WHERE sym_id = %d""", (info['name'], sym_id))
            cursor.execute("""INSERT INTO etf_info VALUES(%s, %s, %s)""", (sym_id, self.__today, json.dumps(info)))
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

    def updatekQuotes(self, symbol, quotes, reset=False):
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
