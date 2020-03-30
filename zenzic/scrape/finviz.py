import os

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from zenzic.data.stockdb import StockDB
from pathlib import Path

class FinViz:
    def __init__(self):
        ublock_path = os.path.join(os.path.dirname(Path(__file__).absolute()), 'chrome-extension', 'uBlock-Origin_v1.24.4.crx')
        # print(ublock_path)
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_extension(ublock_path)
        self.__chrome = webdriver.Chrome(options=chrome_options)

    def getStockInfo(self, symbol):
        url = 'https://finviz.com/quote.ashx?t=' + symbol
        stock_info = {}
        try:
            while True:
                self.__chrome.get(url)
                ticker = self.__chrome.find_element_by_class_name('fullview-ticker')
                if ticker.text == symbol:
                    break
            
            rows =  self.__chrome.find_elements_by_css_selector('.snapshot-table2 tbody tr')
            for row in rows:
                cols = row.find_elements_by_css_selector('td')
                idx = 0
                for col in cols:
                    idx += 1
                    if idx % 2 == 0:
                        stock_info[cols[idx - 2].text.strip()] = col.text.strip()
            
            return stock_info
        except NoSuchElementException:
            return None

if __name__ == "__main__":
    finviz = FinViz()
    stock_db = StockDB()
    symbols = stock_db.getStockSymbols()
    for symbol in symbols:
        stock_info = finviz.getStockInfo(symbol)
        if stock_info:
            stock_db.updateStockInfo(symbol, stock_info)
    del finviz

