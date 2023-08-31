import PySimpleGUI as sg
import os

from image_utils import ImageHelper
from layout import get_layout
from watchdir import watch_directory
from displaydiskdetector import DisplayDiskDetector

def run():
    sg.theme('Black')
    layout = get_layout()
    observer = None

    window = sg.Window('Sélection de répertoire et affichage d\'image', layout, finalize=True, element_justification='c')
    image_helper = ImageHelper()


    window['-IMAGE-'].update(data=image_helper.open_image_base())
    display_detector = DisplayDiskDetector(window, image_helper, image_helper.image_base,309.83)


    running = False
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Quitter':
            if (observer != None):
                observer.stop()
                observer.join()
            break

        if event == "Analyse":
            if running:
                observer.stop()
                observer.join()
                window['-IMAGE-'].update(data=image_helper.open_image_base())   
                window.Element('Analyse').update('Analyze')
                running = False
            else:
                folder_path = values['-FOLDER-']
                if os.path.isdir(folder_path):
                    running = True
                    window.Element('Analyse').update('Stop')
                    window.Element('-RUNNING-').update('Watching Repertory %s' % folder_path)
                    window.Refresh()
                    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.ser'))]
                    if len(image_files) > 0:
                        
                        #window['-IMAGE-'].update(filename=image_file_path)
                        for f in image_files:
                            image_file_path = os.path.join(folder_path, f)
                            display_detector.detect_and_display(image_file_path)
                    observer = watch_directory(folder_path, display_detector.file_call_back, ['ser'])
        
    window.close()


if __name__ == "__main__":
    run()