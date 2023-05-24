from functools import reduce
from time import time

from ModelParser import *
from DocsParser import *
from CommonParser import *


def main(path_to_model=''):
    if path_to_model == '':
        mp = ModelParser(set_all=True)
    else:
        mp = ModelParser(path_to_model, set_all=True)
    
    etp_to_xscades = mp.get_etp_to_xscades()
    xscade_to_operator_tree = mp.get_xscade_to_operator_tree()
    widgets = mp.get_widgets()

    dp = DocsParser(set_all=True)
    req_to_widgets = dp.get_req_to_widgets()
    all_srd_text = dp.get_all_srd_text()
    ds_to_msg = dp.get_ds_to_msg()
    
    if path_to_model == '':
        cp = CommonParser(xscade_to_operator_tree, widgets, req_to_widgets, ds_to_msg, all_srd_text)
    else:
        cp = CommonParser(xscade_to_operator_tree, widgets, req_to_widgets, ds_to_msg, all_srd_text, path_to_model)
    
    while True:
        in_output_or_in_model = input('\nХотите записать итоговые данные только в папку "output" или в саму модель тоже? (введите 1 или 2): ')
        if in_output_or_in_model == '1':
            cp.make_trace_files(etp_to_xscades, xscade_to_operator_tree, write_to_model=False)
            break
        elif in_output_or_in_model == '2':
            cp.make_trace_files(etp_to_xscades, xscade_to_operator_tree, write_to_model=True)
            break
        else:
            print('Введите 1 или 2.')

    return mp, dp, cp


def menu():
    code = ''
    path_to_model = input('Введите путь до модели (".../***_MBD"). Чтобы использовать путь по-умолчанию нажмите Enter:\n')
    print()
    if path_to_model == '':
        output = [ModelParser(), DocsParser(), CommonParser()]
    else:
        output = [ModelParser(path=path_to_model), DocsParser(), CommonParser()]
    started_before = False
    preview = '\nКакие данные вы хотите посмотреть?\n\t1 - Трассировка etp на xscade\n\t2 - Трассировка xscade на переменные\n\t3 - Список виджетов\n\t4 - Дерево вызовов операторов\n\t5 - Трассировка требований на виджеты\n\t6 - Все тексты требований\n\t7 - Трассировка ds на msg\n\t8 - Трассировка не ext на srd\n\t9 - Трассировка ext на srd\n\t10 - Финальная трассировка'
    while True:
        code = input('\nВыбор режима запуска:\n\t1 - Получение файлов трассировки\n\t2 - Посмотреть часть трассировки\n\t3 - Выход\n')
        if code == '1':
            print('\n --- Начало работы ---\n')
            output = list(main(path_to_model))
            print('\nРезультаты находятся в: ' + os.getcwd() + '\\output')
            print('\n --- Работа завершена ---\n')
            started_before = True
        elif code == '2':
            print(preview)
            while True:
                part = input()
                if not part.isdigit():
                    print('Некорректный ввод. Повторите.\n')
                    continue
                limit = int(input('\nВведите лимит вывода данных (используйте -1 чтобы увидеть все данные): '))
                filename = input('Хотите записать данные в файл? Если да - укажите имя файла, если нет - нажмите Enter\n')
                print()
                to_file = True if filename != '' else False

                if limit == -1:
                    limit = None
                
                if part == '1':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '2':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '3':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '4':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '5':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '6':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '7':
                    '''
                    github
                    here is removed code
                    '''
                    break
                elif part == '8':
                    if not started_before:
                        output[0].set_xscade_to_operator_tree()
                        output[0].set_widgets()
                        output[1].set_req_to_widgets()
                        if path_to_model == '':
                            output[2].set_xscade_to_fullpath(output[0].get_xscade_to_operator_tree())
                        else:
                            output[2].set_xscade_to_fullpath(output[0].get_xscade_to_operator_tree(), path_to_model)

                        output[2].set_xscade_to_srd(output[0].get_xscade_to_operator_tree(), output[0].get_widgets(), output[1].get_req_to_widgets())
                    if to_file:
                        output[2].print_data('xscade_to_srd', limit, to_file=True, filename=filename)
                    else:
                        output[2].print_data('xscade_to_srd', limit)
                    break
                elif part == '9':
                    if not started_before:
                        output[0].set_xscade_to_operator_tree()
                        output[1].set_ds_to_msg()
                        output[1].set_req_to_widgets()
                        if path_to_model == '':
                            output[2].set_xscade_to_fullpath(output[0].get_xscade_to_operator_tree())
                        else:
                            output[2].set_xscade_to_fullpath(output[0].get_xscade_to_operator_tree(), path_to_model)
                        output[2].set_ext_to_srd(output[0].get_xscade_to_operator_tree(), output[1].get_ds_to_msg(), output[1].get_all_srd_text())
                    if to_file:
                        output[2].print_data('ext_to_srd', limit, to_file=True, filename=filename)
                    else:
                        output[2].print_data('ext_to_srd', limit)
                    break
                elif part == '10':
                    if not started_before:
                        output[0] = ModelParser(set_all=True)
                        output[1] = DocsParser(set_all=True)
                        output[2].set_xscade_to_fullpath(output[0].get_xscade_to_operator_tree(), path_to_model)
                        output[2].set_xscade_to_srd(output[0].get_xscade_to_operator_tree(), output[0].get_widgets(), output[1].get_req_to_widgets())
                        output[2].set_ext_to_srd(output[0].get_xscade_to_operator_tree(), output[1].get_ds_to_msg(), output[1].get_all_srd_text())
                        output[2].set_common_to_srd()
                        started_before = True
                    if to_file:
                        output[2].print_data('common_to_srd', limit, to_file=True, filename=filename)
                    else:
                        output[2].print_data('common_to_srd', limit)
                    break
                else:
                    print('Некорректный ввод. Повторите.\n')
        elif code == '3':
            return
        else:
            print('Некорректный ввод. Повторите.\n')


if __name__ == '__main__':
    menu()
