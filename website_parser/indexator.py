import pandas as pd
import os
from logger import Logger
import time


class Indexator(object):
    def __init__(self, log=Logger(), basename=r'base_of_links.csv'):
        self.log = log# Logger
        self.log.write_to_log(time.strftime('%c'))
        self.log.write_to_log('Indexator was created')
    
        self.basename = basename
        self.old_links = self.read_from_base()#    same list
        self.new_links = []#    list [[link1, date1], [link2, date2], ..., [linkn, daten]]
        self.bot_links = []#    similar list


    def set_new_links(self, lst):
        self.bot_links = []
        self.log.write_to_log("set_new_links, len(lst) = " + str(len(lst)))
        
        old_links_set = set([i[0] for i in self.old_links])
        new_links_set = set([i[0] for i in lst]) - old_links_set

        if len(new_links_set) == 0:
            print('No new links')
            self.log.write_to_log("No new links")
            return 0
        else:
            self.log.write_to_log("Have some new links")   
            self.new_links = sorted(list(filter(lambda x: x[0] in new_links_set, lst)), reverse=True, key=lambda x: x[1])
            self.__set_bot_links()
            self.__write_to_base()
            return len(self.new_links)


    def __set_bot_links(self):
        '''
        github
        here is removed code
        '''
        return len(self.bot_links)


    def get_bot_links(self):
        return self.bot_links.copy()


    def __write_to_base(self):
        self.log.write_to_log("write_to_base")
        lst = sorted(self.old_links + self.new_links, reverse=True, key=lambda x: x[1])

        df = pd.DataFrame({'links': [i[0] for i in lst], 'date': [i[1] for i in lst], 'simple_date' : [i[2] for i in lst]})
        df.to_csv(self.basename, index=False)
        print("Numder of links in base:", len(df))
        self.log.write_to_log("Numder of links in base:" + str(len(df)))
        
        self.old_links = lst.copy()


    def read_from_base(self):
        self.log.write_to_log("read_from_base")
        if self.basename not in os.listdir(path=os.getcwd()):
            print("There is no base of links")
            self.log.write_to_log("There is no base of links")
            df = pd.DataFrame({'links':[], 'date':[], 'simple_date':[]})
            df.to_csv(self.basename, index=False)
            return []
        else:
            '''
            github
            here is removed code
            '''
            return lst


    def shrink_base(self, num=100000):
        self.log.write_to_log("shrink_base")
        df = pd.read_csv(self.basename)
        print("Numder of links in base: ", len(df))
        self.log.write_to_log("Numder of links in base: " + str(len(df)))
        '''
        github
        here is removed code
        '''
