import os
import datetime
import pymysql as mysql
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from zenzic.scrape.chromeext import uBlockOrigin

LOTTERIES = {
    'mega-millions': {
        'short-name': 'MM',
        'pick_5_class': 'ball',
        'pick_1_class': 'mega-ball'
    },
    'powerball': {
        'short-name': 'PB',
        'pick_5_class': 'ball',
        'pick_1_class': 'powerball'
    },
    'california/superlotto-plus': {
        'short-name': 'CA/SLP',
        'pick_5_class': 'ball',
        'pick_1_class': 'mega-ball'
    }
}

class Lottery:
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_extension(uBlockOrigin())
        self.__chrome = webdriver.Chrome(options=chrome_options)
        self.__conn = mysql.connect(read_default_file="~/my.cnf")

    def __find_elements(self, element, css_selector):
        try:
            return element.find_elements_by_css_selector(css_selector)
        except NoSuchElementException:
            return None

    def sync(self, name, sync_all=False):
        year = datetime.date.today().year
        count = 0

        while True:
            url = 'https://www.lottery.net/' + name + '/numbers/' + str(year)
            year -= 1
            self.__chrome.get(url)
            result_tbl = self.__find_elements(self.__chrome, 'table[class^=\'prizes archive\']')
            if result_tbl:
                count += 1
                rows = self.__find_elements(self.__chrome, 'tbody tr')
                cursor = self.__conn.cursor()
                for row in rows:
                    draw_date = self.__find_elements(row, 'td a')
                    assert len(draw_date) == 1
                    draw_date = datetime.datetime.strptime(draw_date[0].text, r'%A %B %d, %Y')
                    pick_5 = self.__find_elements(row, 'li[class=\'{}\']'.format(LOTTERIES[name]['pick_5_class']))
                    if name == 'california/superlotto-plus':
                        # CA SupperLott Plus doesn't have Mega ball before 06/03/2000.
                        break
                    else:
                        assert len(pick_5) == 5
                    pick_5 = ','.join([n.text for n in pick_5])
                    pick_1 = self.__find_elements(row, 'li[class=\'{}\']'.format(LOTTERIES[name]['pick_1_class']))
                    assert len(pick_1) == 1
                    pick_1 = pick_1[0].text
                    sql = (
                        "INSERT INTO lottery_draws (lottery, draw_date, pick_5, pick_1) "
                        "VALUES(%s, %s, %s, %s) AS new "
                        "ON DUPLICATE KEY UPDATE lottery=new.lottery, draw_date=new.draw_date, pick_5=new.pick_5, pick_1=new.pick_1"
                    )
                    cursor.execute(sql, (LOTTERIES[name]['short-name'], draw_date, pick_5, pick_1))
                self.__conn.commit()
                if not sync_all and count > 0:
                    break
            else:
                # Couldn't find any draw results.
                if sync_all and (count > 0 or year < 1980):
                    break

if __name__ == "__main__":
    lottery = Lottery()
    for name in LOTTERIES.keys():
        lottery.sync(name)
    del lottery
