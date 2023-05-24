from crawler import Crawler, ErrorResponseCode
from indexator import Indexator
from logger import Logger
from bot import Bot
import time
import random


url = [r'https://www.***.ge/en/s/...',
        r'https://www.***.ge/en/s/...',
        r'https://www.***.ge/en/s/...',
        r'https://www.***.ge/en/s/...',
        r'https://www.***.ge/en/s/...',
        r'https://www.***.ge/en/s/...']

bot = Bot(bot_token=r'***', id=***, sale_id=***, support_id=***)

log = Logger(target + '_log.txt')
ind = Indexator(log, target + '_base.csv')
cr = Crawler(url, log)

'''
github
here is removed code
'''
cr.send_request()
'''
github
here is removed code
'''
cr.parse_answer()
'''
github
here is removed code
'''
links = cr.get_links()
'''
github
here is removed code
'''
ind.set_new_links(links)
bot_links = ind.get_bot_links()
'''
github
here is removed code
'''
for j, l in enumerate(sorted(bot_links, key=lambda x: x[1])):
    if len(bot_links) - j <= 30:
        bot.send_link('''
                      github
                      here is removed code
                      ''')
        log.write_to_log(time.strftime('%c') + ' - link is sent to bot')
    print(j, l)
'''
github
here is removed code
'''
