import PySimpleGUI as sg
import os

# Créer la disposition de l'interface graphique
layout = [
    [sg.Text('Sélectionnez un répertoire contenant une image :')],
    [sg.InputText(key='-FOLDER-'), sg.FolderBrowse()],
    [sg.Button('Analyse')],
    [sg.Image(key='-IMAGE-', size=(400, 400))],
    [sg.Button('Quitter')]
]

window = sg.Window('Sélection de répertoire et affichage d\'image', layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Quitter':
        break
    if event == "Analyse":
        folder_path = values['-FOLDER-']
        print(folder_path)
        image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if len(image_files) > 0:
            print("oky")
            image_file_path = os.path.join(folder_path, image_files[0])
            window['-IMAGE-'].update(filename=image_file_path)

window.close()
