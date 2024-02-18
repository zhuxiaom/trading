import os
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from zenzic.data.stockdb import StockDB
from zenzic.scrape.chromeext import uBlockOrigin

class FinViz:
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_extension(uBlockOrigin())
        self.__chrome = webdriver.Chrome(options=chrome_options)

    def getStockInfo(self, symbol):
        url = 'https://finviz.com/quote.ashx?t=' + symbol
        stock_info = {}
        try:
            while True:
                self.__chrome.get(url)
                ticker = self.__chrome.find_element_by_class_name('quote-header_ticker-wrapper_ticker')
                if ticker.text == symbol:
                    break

            row = self.__chrome.find_element_by_css_selector('h2[class^="quote-header_ticker-wrapper_company"] > a')
            assert(row.text.strip()), "Can't find company name!"
            stock_info["Company Name"] = row.text.strip()

            rows = self.__chrome.find_elements_by_css_selector('div.quote-links > div:nth-child(1) > a.tab-link')
            assert (len(rows) == 4), "Only got %s" % (len(rows))
            stock_info["Sector"] = rows[0].text.strip()
            stock_info["Industry"] = rows[1].text.strip()
            stock_info["Country"] = rows[2].text.strip()
            stock_info["Exchange"] = rows[3].text.strip()
            
            rows =  self.__chrome.find_elements_by_css_selector('table.snapshot-table2 .table-dark-row')
            assert (len(rows) in (12, 13)), "Only got %s" % (len(rows))
            for row in rows:
                cols = row.find_elements_by_css_selector('td')
                assert (len(cols) == 12), "Only got %s" % (len(cols))
                idx = 0
                for col in cols:
                    idx += 1
                    if idx % 2 == 0:
                        stock_info[cols[idx - 2].text.strip()] = col.text.strip()
            
            return stock_info
        except TimeoutException:
            print("Retry symbol '%s'." % (symbol))
            return self.getStockInfo(symbol)
        except NoSuchElementException:
            return {}
        except WebDriverException:
            return {}
        
    def close(self):
        self.__chrome.close()
        self.__chrome.quit()

if __name__ == "__main__":
    finviz = FinViz()
    stock_db = StockDB()
    symbols = stock_db.getStockSymbols()
    total = len(symbols)
    cnt = 0
    for symbol in symbols:
        cnt += 1
        if cnt % 100 == 0:
            # Reset Chrome to mitigate out of memory issue.
            finviz.close()
            del finviz
            finviz = FinViz()
        stock_info = finviz.getStockInfo(symbol)
        print("Updating %s information with %d keys. %d more symbols to be updated." % (symbol, len(stock_info), total - cnt))
        stock_db.updateStockInfo(symbol, stock_info)
    del finviz

