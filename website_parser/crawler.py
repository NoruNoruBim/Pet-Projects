import requests
from bs4 import BeautifulSoup
import re
import socks
import socket
# from fake_useragent import UserAgent
from pyuser_agent import UA
from logger import Logger
import time
import random

socks.set_default_proxy(socks.SOCKS5, 'localhost', 9150)
socket.socket = socks.socksocket


class ErrorResponseCode(Exception):
    pass


class Crawler(object):
    def __init__(self, url=[r'https://www.***.ge/en/s/...'], log=Logger()):
        self.root_url = url
        self.__links = []#  [[l, d], [l, d], ...]
        # self.user_agent = [UserAgent().chrome for url in self.root_url]
        self.user_agent = [UA().chrome for url in self.root_url]
        self.session = requests.Session()
        self.session.headers['User-Agent'] = self.user_agent[0]
        
        self.session.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        self.session.headers['Accept-Encoding'] = 'gzip, deflate, br'
        self.session.headers['Accept-Language'] = 'en-US,en;q=0.5'
        self.session.headers['Host'] = 'www.***.ge'
        self.log = log# Logger
        for i in range(10):# here is place of common error (no internet)
            try:
                self.ip = self.check_ip()
                break
            except:
                print('   Switch on internet')
                self.log.write_to_log('   Switch on internet')
                time.sleep(300)#   5 min
                print("!!!    was asleep 300", i, 'time')
                self.log.write_to_log('!!!    was asleep 300')
        else:
            self.ip = self.check_ip()
        self.req = []

        self.log.write_to_log(time.strftime('%c'))
        self.log.write_to_log('Crawler was created')
        self.log.write_to_log('root_url: ' + ', '.join(self.root_url))
        self.log.write_to_log('UserAgent: ' + ', '.join(self.user_agent))
        self.log.write_to_log('ip: ' + self.ip)


    def check_ip(self):
        return requests.get('http://icanhazip.com').text[:-1]


    def send_request(self):#    send request - return html text of page
        '''
        github
        here is removed code
        '''
        return self.req


    def _convert_back(self, number):
        return str((number // 60) % 24) + ':' + str(number % 60)#   h:min


    def __get_cost(self, info):
        return info.find('b', {'class' : 'item-price-usd mr-2'}).text

    def __get_floor(self, info):
        res = info.find('div', {'data-tooltip' : 'Floor'})
        return '?' if res == None else res.span.text.split()[-1]

    def __get_room(self, info):
        res = info.find('div', {'data-tooltip' : 'Number of rooms'})
        return '?' if res == None else res.span.text.split()[-1]

    def __get_bedroom(self, info):
        res = info.find('div', {'data-tooltip' : 'Bedroom'})
        return '?' if res == None else res.span.text.split()[-1]
    
    def __get_meters(self, info):
        res = info.find('div', {'class' : 'item-size'})
        return '?' if res == None else res.text

    def __get_address(self, info):
        res = info.find('div', {'class' : 'address'})
        return '?' if res == None else res.text


    def _get_key(self, info):# function to get key for sorting in date order
        months = {'Apr' : 4, 'May' : 5, 'Jun' : 6, 'Jul' : 7, 'Aug' : 8, 'Sep' : 9, 'Oct' : 10}
        d_m_h_m = info.find('span', {'class' : 'd-block mb-3'}).text.replace(':', ' ').split()

        return ((int(d_m_h_m[0]) + months[d_m_h_m[1][:3]] * 30) * 24 + int(d_m_h_m[2])) * 60 + int(d_m_h_m[3])


    def parse_answer(self):#    get html text - return clear list of links from html
        self.log.write_to_log("parse_answer")

        for url, req in zip(self.root_url, self.req):
            soup = BeautifulSoup(req.text, "lxml")# translate to text

            tags = soup.find_all('a', href=re.compile(r'https://www.***.ge/en/pr'\
                                                        + r'/[a-zA-Z0-9\/\.\-]*'))

            tags = sorted(tags, reverse=True, key=self._get_key)

            '''
            github
            here is removed code
            '''


    def get_links(self, count=-1):
        self.log.write_to_log("get_links")
        return self.__links[:count].copy()
