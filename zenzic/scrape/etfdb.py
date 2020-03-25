import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from zenzic.data.stockdb import StockDB

class EtfDb:
    def __init__(self):
        self.__chrome = webdriver.Chrome()
        self.__sym_info = {}

    def fetch_etf_symbols(self):
        for ch in range(65, 91):
            url = 'https://etfdb.com/alpha/' + chr(ch) + '/'
            self.__chrome.get(url)
            while True:
                rows = self.__chrome.find_elements_by_xpath('//*[@id="etfs"]/tbody/tr')
                for row in rows:
                    elmts = row.find_elements_by_xpath('./*')
                    # print(':'.join([x.text for x in elmts]))
                    self.__sym_info[elmts[0].text.strip()] = {
                        'name': elmts[1].text.strip(),
                        'category': elmts[2].text.strip()
                    }

                try:
                    self.__chrome.find_element_by_css_selector('.page-next.disabled')
                    break
                except NoSuchElementException:
                    next_btn = self.__chrome.find_element_by_css_selector('.page-next a:first-child')
                    ActionChains(self.__chrome).move_to_element(next_btn).click().perform()
                    for row in rows:
                        WebDriverWait(self.__chrome, 10).until(EC.staleness_of(row))
        return self.__sym_info

    def fetch_etf_profile(self, symbol):
        pass

if __name__ == "__main__":
    stock_db = StockDB()
    etf_db = EtfDb()
    symbols = etf_db.fetch_etf_symbols()
    for sym in symbols.keys():
        stock_db.updateEtfDb(sym, symbols[sym])
    del etf_db