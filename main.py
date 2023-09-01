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

    window = sg.Window('Astro mosaic helper [easyastro]', layout, finalize=True, element_justification='c')
    image_helper = ImageHelper()

    mosaic_type=0
    window['-IMAGE-'].update(data=image_helper.open_image_base(mosaic_type))
    


    running = False
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Quitter':
            if (observer != None):
                observer.stop()
                observer.join()
                display_detector.stop()
            break

        if event=='-OPTION_SUN-' and not running:
            mosaic_type=1
            window['-IMAGE-'].update(data=image_helper.open_image_base(mosaic_type))
        if event=='-OPTION_MOON-' and not running:
            mosaic_type=0
            window['-IMAGE-'].update(data=image_helper.open_image_base(mosaic_type))


        if event == '-UPDATE MINIATURE-':
            window['-IMAGE-'].update(data=image_helper.cv2_to_sg(display_detector.base_image))
            window.extend_layout(window['thumb'],image_helper.min_to_sg_grid(display_detector.thumbnails[values[event]]))
            window['thumb'].Widget.update() 
            window['thumb'].contents_changed()
            window.Refresh()

        if event == '-NEW TASK-':
            window.Element('-TASKING-').update('Working file: %s' % values[event])

        if event == "Analyse":
            if running:
                observer.stop()
                observer.join()
                display_detector.stop()

                window['-IMAGE-'].update(data=image_helper.open_image_base(mosaic_type))   
                window.Element('Analyse').update('Analyze')
                running = False
            else:
                folder_path = values['-FOLDER-']
                if os.path.isdir(folder_path):
                    running = True
                    window.Element('Analyse').update('Stop')
                    window.Element('-RUNNING-').update('Watching Repertory %s' % folder_path)
                    window.Refresh()
                    display_detector = DisplayDiskDetector(window, image_helper, image_helper.image_base,309.83)
                    image_files = [file for file in os.listdir(folder_path) if ImageHelper.is_image_supported_format(file)]
                    if len(image_files) > 0:                
                        for f in image_files:
                            image_file_path = os.path.join(folder_path, f)
                            display_detector.detect_and_display(image_file_path)
                    observer = watch_directory(folder_path, display_detector.file_call_back, ['ser'])
        
    window.close()


if __name__ == "__main__":
    run()