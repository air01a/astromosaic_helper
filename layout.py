import PySimpleGUI as sg

def get_layout():
    return [
        [sg.Text('Sélectionnez un répertoire contenant une image :')],
        [sg.InputText(key='-FOLDER-'), sg.FolderBrowse()],
        [sg.Radio('Sun', "TYPE", key='-OPTION1-', default=True),sg.Radio('Moon', "TYPE", key='-OPTION2-', default=False),sg.Radio('DeepSky', "TYPE", key='-OPTION3-', default=False)],
        [sg.Button('Analyse')],
        [sg.Image(key='-IMAGE-', size=(640, 480))],
        [sg.Button('Quitter')]
    ]