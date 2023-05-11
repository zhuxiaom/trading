import os
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from zenzic.data.stockdb import StockDB
from zenzic.scrape.chromeext import uBlockOrigin
from pathlib import Path

class EtfDb:
    def __init__(self):
        # chrome_options = webdriver.ChromeOptions()
        # prefs = {"profile.managed_default_content_settings.images": 2}
        # chrome_options.add_experimental_option("prefs", prefs)
        # chrome_options.add_extension(uBlockOrigin())
        # self.__chrome = webdriver.Chrome(options=chrome_options)
        self.__chrome = webdriver.Chrome()
        self.__sym_info = {}

    def fetchEtfSymbols(self):
        for ch in range(65, 91):
            url = 'https://etfdb.com/alpha/' + chr(ch) + '/'
            self.__chrome.get(url)
            while True:
                rows = self.__chrome.find_elements_by_xpath('//*[@id="etfs"]/tbody/tr')
                print('Found %d ETFs in the page.' % (len(rows)))
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
                    while True:
                        next_btn = self.__chrome.find_element_by_css_selector('.page-next a:first-child')
                        ActionChains(self.__chrome).move_to_element(next_btn).click().perform()
                        try:
                            for row in rows:
                                WebDriverWait(self.__chrome, 10).until(EC.staleness_of(row))
                            break
                        except TimeoutException:
                            print('Retry next page ...')

        return self.__sym_info

    def fetchEtfProfile(self, symbol):
        pass

if __name__ == "__main__":
    stock_db = StockDB()
    etf_db = EtfDb()
    symbols = etf_db.fetchEtfSymbols()
    for sym in symbols.keys():
        stock_db.updateEtfDb(sym, symbols[sym])
    del etf_db