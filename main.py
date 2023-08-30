import PySimpleGUI as sg
import os
from diskdetector import DiskDetector
from image_utils import ImageHelper
from layout import get_layout



def run():
    # Créer la disposition de l'interface graphique
    sg.theme('Black')
    layout = get_layout()

    window = sg.Window('Sélection de répertoire et affichage d\'image', layout, finalize=True, element_justification='c'
    
    )
    image_helper = ImageHelper()


    window['-IMAGE-'].update(data=image_helper.open_image_base())

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Quitter':
            break

        if event == "Analyse":
            folder_path = values['-FOLDER-']
            image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.ser','.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            if len(image_files) > 0:
                image_file_path = os.path.join(folder_path, image_files[0])
                window['-IMAGE-'].update(filename=image_file_path)

    window.close()


if __name__ == "__main__":
    run()