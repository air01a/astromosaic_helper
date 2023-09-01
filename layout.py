import PySimpleGUI as sg

def get_layout():
    return [
        [sg.Text('Sélectionnez un répertoire contenant une image :')],
        [sg.InputText(key='-FOLDER-'), sg.FolderBrowse()],
        [sg.Radio('Moon', "TYPE", key='-OPTION_MOON-', default=True, enable_events=True),sg.Radio('Sun', "TYPE", key='-OPTION_SUN-', default=False,  enable_events=True)],
        [sg.Button('Analyse')],
        [sg.Text('',  text_color='green', key='-RUNNING-'), sg.Text('',  text_color='blue', key='-TASKING-')],
        [sg.Image(key='-IMAGE-', size=(640, 480)),sg.Column([], key='thumb',scrollable=True, vertical_scroll_only=True, size=(200, 480))],
        [sg.Button('Quitter')]
    ]