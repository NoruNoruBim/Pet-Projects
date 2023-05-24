from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Label
import tkinter.filedialog
import os
import re
import DB_BE


def check_and_parse(layers_list):
    if layers_list == 'all':
        return [i for i in range(1, 17)]
    elif layers_list == '':
        return []
    else:
        pattern = re.compile(r'[\d ,]+')
        assert pattern.findall(layers_list)[0] == layers_list, 'Wrong list of layers. Correct sample: "1, 2, 3"'
        layers_list = list(map(lambda x: int(x), layers_list.replace(' ', '').split(',')))
        assert all(map(lambda x: type(x) == int and 0 <= x <= 16, layers_list)), 'Use correct layer numbers [0; 16]'
        return layers_list


def ChooseDBFile():
    global path_to_db
    path_to_db = tkinter.filedialog.askopenfilename(initialdir=os.getcwd(), filetypes = [('*.abn', '.abn')])
    if path_to_db != '':
        text_db.set(path_to_db)
        extract['state'] = 'normal'
        choose_excel['state'] = 'normal'


def ChooseExcelFile():
    if choose_excel['state'] == 'normal':
        global path_to_excel
        path_to_excel = tkinter.filedialog.askdirectory(initialdir=os.getcwd())
        if path_to_excel != '':
            text_excel.set(path_to_excel)
            choose_copy['state'] = 'normal'


def ChooseCopyFile():
    if choose_copy['state'] == 'normal':
        global path_to_copy
        path_to_copy = tkinter.filedialog.asksaveasfilename(initialdir=os.getcwd(), filetypes = [('*.abn', '.abn')])
        if path_to_copy != '':
            path_to_copy = path_to_copy if path_to_copy.endswith('.abn') else path_to_copy + '.abn'
            text_copy.set(path_to_copy)
            build['state'] = 'normal'


def ExtractDB():
    if extract['state'] == 'normal':
        global db, layers_list
        layers_list = check_and_parse(text_layers.get())
        
        db = DB_BE.DataBase(path_to_db, layers_list)#  Always read Header
        db.write_to_excel(layers_list)


def BuildDB():
    if build['state'] == 'normal':
        global db, layers_list
        layers_list = check_and_parse(text_layers.get())
    
        db = DB_BE.DataBase(path_to_db, [i for i in range(1, 17)])
        db.modify(path_to_excel + '/', layers_list)
        db.write_to_file(path_to_copy, [i for i in range(0, 17)])


root = Tk()
root.geometry('700x470+250+150')
root.title('DB_BE')
root.configure(background='ivory2')

label_choose_db = Label(text="Выберите БД:", font='Helvetica 16 bold', background = "ivory2")
label_choose_db.place(x = 20, y = 8)

choose_dbFrame = Frame(root, height = 40, bg = "gray")
choose_dbFrame.place(x = 200, y = 10)

choose_db = Button(choose_dbFrame, text = 'Выбрать', command=ChooseDBFile)
choose_db.pack(side = "left", fill = 'x')

text_db = StringVar()
inp_db = Entry(choose_dbFrame, width = 70, state='readonly', textvariable=text_db)
inp_db.pack(side = 'left', fill = "both", expand = False)


label_choose_layers = Label(text="Выберите слои:", font='Helvetica 16', background = "ivory2")
label_choose_layers.place(x = 20, y = 50)

choose_layersFrame = Frame(root, height = 40, bg = "gray")
choose_layersFrame.place(x = 200, y = 55)

text_layers = StringVar()
text_layers.set('all')
inp_layers = Entry(choose_layersFrame, textvariable=text_layers)
inp_layers.pack(side = 'left', fill = "both", expand = False)

####################################################################################

label_extract = Label(text="Extract", font='Helvetica 16 bold', background = "ivory2")
label_extract.place(x = 20, y = 100)

extractFrame = Frame(root, height = 30, width = 400, bg = "gray")
extractFrame.place(x = 100, y = 140)

extract = Button(extractFrame, text = 'Распаковать', bg="#FF8000", state='disabled', command=ExtractDB)
extract.pack(side = "left", fill = 'x')

####################################################################################

label_build = Label(text="Build", font='Helvetica 16 bold', background = "ivory2")
label_build.place(x = 20, y = 200)

label_choose_excel = Label(text="Выберите папку с excel:", font='Helvetica 16', background = "ivory2")
label_choose_excel.place(x = 20, y = 240)

choose_excelFrame = Frame(root, height = 40, bg = "gray")
choose_excelFrame.place(x = 290, y = 240)

choose_excel = Button(choose_excelFrame, text = 'Выбрать', state='disabled', command=ChooseExcelFile)
choose_excel.pack(side = "left", fill = 'x')

text_excel = StringVar()
inp_excel = Entry(choose_excelFrame, width = 55, state='readonly', textvariable=text_excel)
inp_excel.pack(side = 'left', fill = "both", expand = False)


label_copy = Label(text="Выберите как сохранить:", font='Helvetica 16', background = "ivory2")
label_copy.place(x = 20, y = 280)

choose_copyFrame = Frame(root, height = 40, bg = "gray")
choose_copyFrame.place(x = 290, y = 280)

choose_copy = Button(choose_copyFrame, text = 'Выбрать', state='disabled', command=ChooseCopyFile)
choose_copy.pack(side = "left", fill = 'x')

text_copy = StringVar()
inp_copy = Entry(choose_copyFrame, width = 55, state='readonly', textvariable=text_copy)
inp_copy.pack(side = 'left', fill = "both", expand = False)


buildFrame = Frame(root, height = 40, bg = "gray")
buildFrame.place(x = 100, y = 320)

build = Button(buildFrame, text = 'Создать новую БД', bg="#FF8000", state='disabled', command=BuildDB)
build.pack(side = "left", fill = 'x')

####################################################################################

infoFrame = Frame(root, height = 40, bg = "gray")
infoFrame.place(x = 500, y = 400)
text = 'Сначала выберите основной файл БД.\n\
Теперь вы можете либо распаковать БД, либо изменить и создать новую БД.\n\
Номера слоёв указываются через запятую, например "1, 2, 3". "all" - означает все слои.\n\
После нажатия кнопки необходимо подождать пока она не отожмется обратно (завершится выбранный процесс).\n\
Если участвует 15й слой, то ждать придется долго.'
info = Button(infoFrame, text = "Справка", height = 2, width = 10, bg="#CCE5FF",\
                command = lambda : messagebox.showinfo("Справка", text))
info.pack(side = "left", fill = 'x')

root.mainloop()
