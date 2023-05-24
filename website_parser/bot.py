import telebot
import time


class Bot(object):
    def __init__(self, bot_token=r'***', id=***, sale_id=***, support_id=***):
        self.bot = telebot.TeleBot(bot_token)
        self.main_id = id
        self.sale_id = sale_id
        self.support_id = support_id


    def send_link(self, link, support=False, sale=False):
        '''
        github
        here is removed code
        '''
