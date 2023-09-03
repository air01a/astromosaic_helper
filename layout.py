import PySimpleGUI as sg
import tkinter as tk

def get_window_size():
    tk_root = tk.Tk()
    screen_width = tk_root.winfo_screenwidth()
    screen_height = tk_root.winfo_screenheight()
    dpi = tk_root.winfo_fpixels('1i')
    screen_height = int(screen_height/(dpi/96))
    screen_width = int(screen_width/(dpi/96))
    tk_root.destroy()  
    return(screen_height,screen_width)

def get_layout():

    return [
        [sg.Text('Sélectionnez un répertoire contenant une image :')],
        [sg.InputText(key='-FOLDER-'), sg.FolderBrowse()],
        [sg.Radio('Moon', "TYPE", key='-OPTION_MOON-', default=True, enable_events=True),sg.Radio('Sun', "TYPE", key='-OPTION_SUN-', default=False,  enable_events=True)],
        [sg.Button('Analyse')],
        [sg.Text('',  text_color='green', key='-RUNNING-'), sg.Text('',  text_color='blue', key='-TASKING-')],
        [sg.Image(key='-IMAGE-', size=(640, 480)),sg.Column([], key='thumb',scrollable=True, vertical_scroll_only=True, size=(200, 480))],

    ]