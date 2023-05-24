from tqdm import tqdm
import os
from bs4 import BeautifulSoup 
from copy import deepcopy
import re


#   parse model (.etp/.xscade/.sdy xml files)
class ModelParser:
    # __etps                    - list of paths to .etp files
    # __etp_to_xscades          - dict, where
    #                               key = path to .etp file,
    #                               value = list of references to .xscade files
    # __xscade_to_operator_tree - dict, where
    #                               key = path of .xscade file (uniq),
    #                               value = dict, where
    #                                   key = operator (str)
    #                                   value = dict like:
    #                                       {input : [...], output : [...], data : [...]}
    #                               input and output here is dicts with some information from tag in xml
    # __widgets                 - list (all info about mapping items in .sdy), where
    #                               each elem is a dict, where
    #                                   keys = 'inputs', 'output'
    #                                   values = attrs of this tags in .sdy
    # __xscade_call_tree        - dict, where
    #                               key - name of xscade
    #                               value - dict, where
    #                                   key - name of operator (single for each xscade)
    #                                   value - list of called operators
    # __path                    - path to model directory
    def __init__(self, path=r'***', set_all=False):        
        self.__path = path
        self.__etps = None
        self.__etp_to_xscades = None            
        self.__xscade_to_operator_tree = None
        self.__widgets = None
        self.__xscade_call_tree = None
        if set_all:
            self.set_etps()
            self.set_etp_to_xscades()            
            self.set_xscade_to_operator_tree()
            self.set_widgets()
            self.set_xscade_call_tree()


    # create list like:
    # [path_to_etp1, path_to_etp2, ..., path_to_etp37]
    def __find_all_etps(self):
        etps = []
        tq = tqdm(total=37)
        for i in os.walk(self.__path):#     go from root-folder
            for j in i[2]:
                if '.etp' in j:#                                    find all .etp files
                    etps += [i[0] + '\\' + j]#                      store path (path_to_folder + '\\' + name)
                    tq.update(1)
        tq.close()
        return etps


    # input  - path to .etp file like 'D:\\***\\***_sources\\***\\***\\***\\***.etp'
    # output - all needed links to .xscade files from .etp file
    def __find_xscades(self, etp_file):
        with open(etp_file, 'r') as f:#         get text data from .etp
            text = f.read()
        soup = BeautifulSoup(text, "lxml")#     use xml parser

        all_xscade = soup.find_all('fileref')#    get ALL .xscade files
        #   then filter and store only needed
        needed_xscade = list(filter(lambda x: x.parent.parent.has_attr('extensions') and\
                                                x.parent.parent['extensions'] == 'xscade;scade' or\
                                            x.parent.has_attr('extensions') and\
                                                x.parent['extensions'] == 'xscade;scade', all_xscade))
        needed_xscade = [i['persistas'] for i in needed_xscade]
        return needed_xscade


    # set xscade_to_operator_tree structure using sub-function __create_operator_tree()
    def __extract_xscade_to_operator_tree(self):
        print('please wait...')
        watched = set()
        xscade_to_operator_tree = {}
        tq = tqdm(total=406)
        for etp in self.__etp_to_xscades:
            for xscade in self.__etp_to_xscades[etp]:
                '''
                github
                here is removed code
                '''
                    tq.update(1)
        tq.close()
        return xscade_to_operator_tree


    # sub-function for '__create_operator_tree()'
    def __extract_var_info(self, operator, tag_name):
        tag = operator.find(tag_name)
        if tag == None:
            return ['None']
        elems_to_find = 'variable' if tag_name != 'data' else 'equation'
        elems = tag.find_all(elems_to_find)

        elem_info = []
        for elem in elems:
            if tag_name != 'data':
                elem_info += [(elem['name'],\
                               elem.find('typeref')['name'],\
                               elem.find('ed:variable')['oid'])]
            else:
                '''
                github
                here is removed code
                '''
        
        return elem_info


    # input  - path to .xscade file
    # output - tree
    def __create_operator_tree(self, xscade_file):
        with open(xscade_file, 'r', encoding='ansi') as f:#         get text data from .etp
            text = f.read()
        soup = BeautifulSoup(text, "lxml")#     use xml parser
        
        operator = soup.find('operator')#   get all operators in .xscade file
        operator_to_in_out = {}
        #                                   make dict (tree) from important info
        if operator != None and operator.has_attr('kind'):
            operator_to_in_out[operator['name']] = {'inputs'  : self.__extract_var_info(operator, 'inputs'),\
                                                    'outputs' : self.__extract_var_info(operator, 'outputs'),\
                                                    'data'    : self.__extract_var_info(operator, 'data'),\
                                                    'op_id'   : operator.find('ed:operator', {'oid' : re.compile("\![a-zA-Z\/0-9]+")})['oid']}
        
        return operator_to_in_out


    #   sub-function for '__extract_xscade_call_tree()'
    #   input  - one xscade file (path)
    #   output - call tree to this xscade
    def __create_call_tree(self, xscade_file):
        with open(xscade_file, 'r', encoding='ansi') as f:#         get text data from .etp
            text = f.read()
        soup = BeautifulSoup(text, "lxml")#     use xml parser

        operator = soup.find('operator')#   get all operators in .xscade file
        operator_to_others = {}
        #                                   make dict (tree) from important info
        if operator != None:
            calls = operator.find_all('callexpression')
            inner_operators = []
            for call in calls:
                operator_names = call.find_all('operatorref')
                for name in operator_names:
                    inner_operators = sorted(list(set(inner_operators + [name['name'].split(':')[-1]])))
            operator_to_others[operator['name']] = inner_operators
        return operator_to_others


    #   generate operators call tree: 
    #                   dict where keys - xscades and their operators
    #                              values - called operators)
    def __extract_xscade_call_tree(self):
        print('please wait, make operators call tree...')
        watched = set()
        xscade_call_tree = {}
        tq = tqdm(total=406)
        '''
        github
        here is removed code
        '''
        tq.close()
        return xscade_call_tree
        

    # set-method
    def set_etps(self):
        if self.__etps == None:
            self.__etps = self.__find_all_etps()


    # set-method
    def set_etp_to_xscades(self):
        if self.__etp_to_xscades == None:
            if self.__etps == None:
                self.set_etps()
            self.__etp_to_xscades = {}
            tq = tqdm(total=len(self.__etps))
            '''
            github
            here is removed code
            '''
            tq.close()


    # set-method
    def set_xscade_to_operator_tree(self):
        if self.__xscade_to_operator_tree == None:
            if self.__etp_to_xscades == None:
                self.set_etp_to_xscades()
            self.__xscade_to_operator_tree = self.__extract_xscade_to_operator_tree()


    # set-method
    def set_widgets(self):
        if self.__widgets == None:
            self.__widgets = self.__find_all_widgets()


    # set-method
    def set_xscade_call_tree(self):
        if self.__xscade_call_tree == None:
            '''
            github
            here is removed code
            '''


    # get-method
    def get_etps(self):
        return self.__etps.copy()


    # get-method
    def get_etp_to_xscades(self):
        return deepcopy(self.__etp_to_xscades)


    # get-method
    def get_xscade_to_operator_tree(self):
        return deepcopy(self.__xscade_to_operator_tree)


    # get-method
    def get_widgets(self):
        return deepcopy(self.__widgets)


    # get-method
    def get_xscade_call_tree(self):
        return deepcopy(self.__xscade_call_tree)


    # extract all widgets from .sdy file (in tags 'mappingitem' - 'output' and 'inputs')
    def __find_all_widgets(self):
        with open(self.__path + '\\ROOT\\***_ROOT.sdy', 'r', encoding='ansi') as f:
            text = f.read()
        soup = BeautifulSoup(text, "lxml")#     use xml parser
        
        mapping_items = soup.find_all('mappingitem')
        widget_info = []
        
        tq = tqdm(total=len(mapping_items))
        for item in mapping_items:
            widget_info += [{'output' : list(item.find('output').children)[1].attrs,
                             'inputs' : list(item.find('inputs').children)[1].attrs}]
            tq.update()
        tq.close()
        return widget_info


    # key   - which field we want to print
    # limit - max number of elements which will be printed
    def print_data(self, key, limit=None, to_file=False, filename='file.txt'):#    print-method
        if key == 'etps':
            '''
            github
            here is removed code
            '''
        elif key == 'etp_to_xscades':
            '''
            github
            here is removed code
            '''
        elif key == 'xscade_to_operator_tree':
            '''
            github
            here is removed code
            '''
        elif key == 'widgets':
            '''
            github
            here is removed code
            '''
        elif key == 'xscade_call_tree':
            '''
            github
            here is removed code
            '''
        else:
            print("- Невозможно вывести данные. Используйте следующие ключи: 'etps', 'etp_to_xscades', 'xscade_to_operator_tree', 'widgets'\n")
        
        if to_file:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(s)
        else:
            print(s)
