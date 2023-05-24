from collections import defaultdict
from tqdm import tqdm
from time import time

'''
Here locates some structures (read requerments)
Also here locates some functions, dictionaries and constants which help to work with data
'''

ORDER = 'big'

LAYER_TO_NUMBER = {
        '''
        github
        here is removed code
        '''}

TEMPLATES_FOLDER_NAME = "templates/"
TEMPLATES_FILES_NAMES = {
        '''
        github
        here is removed code
        '''
}


def show(func):
    def sub_func(*args):
        s = ' '.join(func.__name__.split('_')).strip(' ').capitalize()
        print('  ' + s + ' - started\n')
        start = time()
        func(*args)
        stop = time()
        print('  ' + s + ' - done!\n')
        print('%.4f sec' % (stop-start))
        print('-' * 20, end='\n\n')
    return sub_func


def decrypt(b, sign=False):
    return int.from_bytes(b, ORDER, signed=sign)


def encrypt(s, length, sign=False):
    if s == None:
        return b'\x00' * (length)
    elif isinstance(s, str):
        assert length >= (len(s) - 2) // 2 and s.startswith('0x') and len(s) % 2 == 0, 'Invalid hex format in excel'
        res = b''
        for i in range(2, len(s), 2):
            res += int(float.fromhex(s[i:i+2])).to_bytes(1, ORDER, signed=False)
        return b'\x00' * (length - (len(s) - 2) // 2) + res
    return s.to_bytes(length, ORDER, signed=sign)


def decrypt_s(b):
    '''
    github
    here is removed code
    '''


def encrypt_s(s, length):
    '''
    github
    here is removed code
    '''

'''
github
here are 1300 removed lines of code
'''

class Layer:
    def __init__(self, db_file, layer_struct, offset, layer_name):
        '''
        github
        here is removed code
        '''


    def write_to_file(self, db_file, offset, layer_name):
        '''
        github
        here is removed code
        '''


    def write_to_excel(self, sheet_header, sheet, sheet_index, layer_name):
        '''
        github
        here is removed code
        '''


    def modify(self, layer_struct, sheet_header, sheet, sheet_index, layer_name):
        '''
        github
        here is removed code
        '''
        return shift_index, shift_data
        
        
LAYER_TO_STRUCT = {
        '''
        github
        here is removed code
        '''}

