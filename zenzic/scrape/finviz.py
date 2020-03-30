from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from zenzic.data.stockdb import StockDB

class FinViz:
    def __init__(self):
        self.__chrome = webdriver.Chrome()

    def getStockInfo(self, symbol):
        url = 'https://finviz.com/quote.ashx?t=' + symbol
        stock_info = {}
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


if __name__ == "__main__":
    finviz = FinViz()
    stock_db = StockDB()
    symbols = stock_db.getStockSymbols()
    for symbol in symbols:
        stock_info = finviz.getStockInfo(symbol)
        stock_db.updateStockInfo(symbol, stock_info)
    del finviz

