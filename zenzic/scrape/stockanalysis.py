import os
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from zenzic.data.stockdb import StockDB
from zenzic.scrape.chromeext import uBlockOrigin
from pathlib import Path

class StockAnalysis:
    def __init__(self):
        # chrome_options = webdriver.ChromeOptions()
        # prefs = {"profile.managed_default_content_settings.images": 2}
        # chrome_options.add_experimental_option("prefs", prefs)
        # chrome_options.add_extension(uBlockOrigin())
        # self.__chrome = webdriver.Chrome(options=chrome_options)
        self.__chrome = webdriver.Chrome()
        self.__sym_info = {}

    def fetchEtfSymbols(self):
        NUM_OF_COLS = 4

        self.__chrome.get('https://stockanalysis.com/etf/')
        while True:
            cols = self.__chrome.find_elements(By.XPATH, '//th[starts-with(@class, "cursor-pointer")]')
            assert len(cols) == NUM_OF_COLS, f"The num of columns is not {NUM_OF_COLS}."
            rows = self.__chrome.find_elements(By.XPATH, '//td[starts-with(@class, "svelte")]')
            print(f"Found {len(rows)/NUM_OF_COLS} symbols.")
            assert len(rows)%NUM_OF_COLS == 0, f"The num of rows is invalid."
            for i in range(0, len(rows)//NUM_OF_COLS):
                symbol = rows[i*NUM_OF_COLS].text.strip()
                name = rows[i*NUM_OF_COLS+1].text.strip()
                category = rows[i*NUM_OF_COLS+2].text.strip()
                # print(f"symbol: '{symbol}', name: '{name}', category: '{category}'")
                self.__sym_info[symbol] = {
                        'name': name,
                        'category': category
                    }
            next_btn = self.__chrome.find_element(By.XPATH, '//button[@aria-label="Next"]')
            if next_btn.is_enabled():
                ActionChains(self.__chrome).move_to_element(next_btn).click().perform()
            else:
                break
        print(f"Found {len(self.__sym_info)} ETFs in total.")
        return self.__sym_info

    def fetchEtfProfile(self, symbol):
        pass

if __name__ == "__main__":
    stock_db = StockDB()
    etf_db = StockAnalysis()
    symbols = etf_db.fetchEtfSymbols()
    for sym in symbols.keys():
        stock_db.updateEtfDb(sym, symbols[sym])
    del etf_db