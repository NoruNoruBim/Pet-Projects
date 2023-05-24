from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import os
import shutil
from bs4 import BeautifulSoup 
import re


#   concatenate results of ModelParser and DocsParser
class CommonParser:
    def __init__(self, xscade_to_operator_tree=None, widgets=None, req_to_widgets=None, ds_to_msg=None, all_srd_text=None, path= r'D:\***\***_sources\***'):
        self.__xscade_to_srd = None
        self.__ext_to_srd = None
        self.__common_to_srd = None
        self.__xscade_to_fullpath = None
        if xscade_to_operator_tree != None and widgets != None and req_to_widgets != None and ds_to_msg != None and all_srd_text != None:
            self.set_xscade_to_fullpath(xscade_to_operator_tree, path)
            self.set_xscade_to_srd(xscade_to_operator_tree, widgets, req_to_widgets)
            self.set_ext_to_srd(xscade_to_operator_tree, ds_to_msg, all_srd_text)
            self.set_common_to_srd()


    #   link xscade to srd ids
    def _extract_xscade_to_srd(self, xscade_to_operator_tree, widgets, req_to_widgets):
        xscade_to_srd = defaultdict(dict)
        print('please wait, link not ext to srd...')
        tq = tqdm(total=len(xscade_to_operator_tree))
        empty_xscades = []
        for xscade in xscade_to_operator_tree:
            tq.update(1)
            if len(xscade_to_operator_tree[xscade].keys()) == 0:
                empty_xscades += [xscade]
                continue
            
            vars = xscade_to_operator_tree[xscade][list(xscade_to_operator_tree[xscade].keys())[0]]['data']
            redirected = self.__redirection(vars, xscade)
            for widget in widgets:
                if 'widgetident' in widget['output']:
                    var_name1 = widget['inputs']['name'] + widget['inputs']['subelement']
                    widget_id = widget['output']['widgetident']
                elif 'widgetident' in widget['inputs']:
                    var_name1 = widget['output']['name'] + widget['output']['subelement']
                    widget_id = widget['inputs']['widgetident']

                for var_info in vars:
                    '''
                    github
                    here is removed code
                    '''

                    if var_name1 == var_name2:
                        for req in req_to_widgets:
                            if widget_id in req_to_widgets[req]:
                                '''
                                github
                                here is removed code
                                '''
        tq.close()
        
        return xscade_to_srd


    #   extract all 'ext in/out' variables from .xscades (optional all similar, not only ext)
    def _extract_ext_to_srd(self, xscade_to_operator_tree, ds_to_msg, all_srd_text):
        ext_to_srd = defaultdict(dict)
        print('please wait, link ext to srd...')
        tq = tqdm(total=len(xscade_to_operator_tree))
        empty_xscades = []
        
        
        words_from_all_req = defaultdict(set)
        for srd in all_srd_text:
            for raw_word in all_srd_text[srd].split():
                clear_word = raw_word.strip(',. ()[]=-<:;').split('.')[0].lower()
                interval = re.search(r'\<[0-9]+\-[0-9]+\>', clear_word)
                if interval != None and 'ds' == clear_word[:2]:
                    interval = interval.group(0)[1:-1].split('-')
                    '''
                    github
                    here is removed code
                    '''
                    continue
                words_from_all_req[srd].add(clear_word)
        
        for xscade in xscade_to_operator_tree:
            tq.update(1)
            if len(xscade_to_operator_tree[xscade].keys()) == 0:
                empty_xscades += [xscade]
                continue

            in_out = xscade_to_operator_tree[xscade][list(xscade_to_operator_tree[xscade].keys())[0]]['inputs'] +\
                    xscade_to_operator_tree[xscade][list(xscade_to_operator_tree[xscade].keys())[0]]['outputs']

            vars = xscade_to_operator_tree[xscade][list(xscade_to_operator_tree[xscade].keys())[0]]['data']
            types = {}
            for i in range(len(vars)):
                for in_out_info in in_out:
                    if '_L' not in vars[i][0]:
                        if vars[i][0] == in_out_info[0]:
                            types[i] = in_out_info[1]
                            break
                    elif '_L' not in vars[i][1]:
                        if vars[i][1] == in_out_info[0]:
                            types[i] = in_out_info[1]
                            break
                    else:
                        continue
            
            redirected = self.__redirection(vars, xscade)
            for srd in all_srd_text:
                words_from_req = words_from_all_req[srd]
                
                hidden_msg_in_req = set()
                for ds in ds_to_msg:
                    if ds.lower() in words_from_req:
                        hidden_msg_in_req.add(ds_to_msg[ds].lower())

                for i in range(len(vars)):
                    '''
                    github
                    here is removed code
                    '''
        tq.close()
        
        return ext_to_srd


    #   concatenate __xscade_to_srd and __ext_to_srd
    def _extract_common_to_srd(self):
        common_to_srd = deepcopy(self.__xscade_to_srd)
        for i in self.__ext_to_srd:
            for j in self.__ext_to_srd[i]:
                if j in common_to_srd[i]:
                    common_to_srd[i][j][0] = list(set(common_to_srd[i][j][0] + self.__ext_to_srd[i][j][0]))
                else:
                    common_to_srd[i][j] = self.__ext_to_srd[i][j]
        
        with open('input/inner_reqs.txt', 'r') as f:
            text = f.read()
        inner = {i.split(':')[0] : i.split(':')[1].split(',') for i in text.split()}
        
        '''
        github
        here is removed code
        '''
        
        return common_to_srd


    def _extract_xscade_to_fullpath(self, xscade_to_operator_tree, path):
        xscade_to_fullpath = {}
        project_items = list(os.walk(path))
        for xscade in xscade_to_operator_tree:
            for item in project_items:
                if xscade.lower() in [i.lower() for i in item[2]]:
                    xscade_to_fullpath[xscade] = item[0] + '\\' + xscade
        return xscade_to_fullpath


    #   redirect oid of variables to their diagrams
    def __redirection(self, vars, xscade):
        xscade = self.__xscade_to_fullpath[xscade]
            
        with open(xscade, 'r', encoding='utf-8') as f:
            text = f.read()

        soup = BeautifulSoup(text, 'lxml')
        root_diagram = soup.find('ed:operator')
        sub_diagrams = root_diagram.find_all('netdiagram')

        sub_diagrams_name_oid = []
        for i in sub_diagrams:
            sub_diagrams_name_oid += [(i['name'], i['oid'])]

        result = {}
        for var_info in vars:
            not_found = True
            for sub_diagram in sub_diagrams:
                if sub_diagram.find('equationge', {'presentable' : var_info[2]}) != None:
                    result[var_info[2]] = (sub_diagram['oid'], sub_diagram['name'])
                    not_found = False
                    break
            if not_found:
                result[var_info[2]] = (root_diagram['oid'], 'main diagram')

        return result
    
    
    #   just make .trace files (final)
    def make_trace_files(self, etp_to_xscades, xscade_to_operator_tree, write_to_model=False):
        if 'output' not in os.listdir():
            os.mkdir('output')
        else:
            for file in list(os.walk('output'))[0][2]:
                os.remove(os.getcwd() + '/output/' + file)
        for etp in etp_to_xscades:
            not_empty = False
            filename = 'output/' + etp.split('\\')[-1][:-3] + 'trace'
            text = '<?xml version="1.0" encoding="UTF-8"?>\n<traceability>\n'
            for xscade in etp_to_xscades[etp]:
                xscade_short = xscade.split('\\')[-1]
                if xscade_short in self.__common_to_srd:
                    not_empty = True
                    '''
                    github
                    here is removed code
                    '''
            text += '\n</traceability>\n'
            if not_empty:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                if write_to_model:
                    with open(etp[:-3] + 'trace', 'w', encoding='utf-8') as f:
                        f.write(text)
                    types_file_name = list(filter(lambda x: x[-6:] == '.types', os.listdir('input')))[0]
                    try:
                        shutil.copy('input\\' + types_file_name, '\\'.join(etp.split('\\')[:-1]))
                    except FileNotFoundError as e:
                        print(e)
                        print("It's warning, not error. If you want to put '.types' file into needed folders,\
                                you have to put this file into 'input' and restart program.\n.trace files were written correctly.")


    #   set-method
    def set_xscade_to_srd(self, xscade_to_operator_tree, widgets, req_to_widgets):
        if self.__xscade_to_srd == None:
            self.__xscade_to_srd = self._extract_xscade_to_srd(xscade_to_operator_tree, widgets, req_to_widgets)


    #   set-method
    def set_ext_to_srd(self, xscade_to_operator_tree, ds_to_msg, all_srd_text):
        if self.__ext_to_srd == None:
            self.__ext_to_srd = self._extract_ext_to_srd(xscade_to_operator_tree, ds_to_msg, all_srd_text)


    #   set-method
    def set_common_to_srd(self):
        if self.__common_to_srd == None:
            self.__common_to_srd = self._extract_common_to_srd()


    #   set-method
    def set_xscade_to_fullpath(self, xscade_to_operator_tree, path=r'D:\***\***_sources\***'):
        if self.__xscade_to_fullpath == None:
            self.__xscade_to_fullpath = self._extract_xscade_to_fullpath(xscade_to_operator_tree, path)


    #   get-method
    def get_xscade_to_srd(self):
        return deepcopy(self.__xscade_to_srd)


    #   get-method
    def get_ext_to_srd(self):
        return deepcopy(self.__ext_to_srd)


    #   get-method
    def get_common_to_srd(self):
        return deepcopy(self.__common_to_srd)


    #   get-method
    def get_xscade_to_fullpath(self):
        return deepcopy(self.__xscade_to_fullpath)


    # key   - which field we want to print
    # limit - max number of elements which will be printed
    def print_data(self, key, limit=None, to_file=False, filename='file.txt'):#    print-method
        if key == 'xscade_to_srd':
            '''
            github
            here is removed code
            '''
        elif key == 'ext_to_srd':
            '''
            github
            here is removed code
            '''
        elif key == 'common_to_srd':
            '''
            github
            here is removed code
            '''
        else:
            print("- Can't print data. Use folowing args: 'xscade_to_srd', 'ext_to_srd', 'common_to_srd'\n")

        if to_file:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(s)
        else:
            print(s)
