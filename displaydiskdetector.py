import cv2
from serreader import SerReader
from diskdetector import DiskDetector
import numpy as np
import time
import os


class DisplayDiskDetector():
    def __init__(self, window, image_helper, base_image, new_r):
        self.new_r = new_r
        self.base_image = base_image
        self.window = window
        self.image_helper = image_helper

        (base_w,base_h) = (self.base_image.shape[1], self.base_image.shape[0])
        self.center_x = int(base_w/2)
        self.center_y = int(base_h/2)
    

    def detect_and_display(self, path):
        image_copy = self.base_image.copy()
        ser_reader = SerReader(path)
        frame = ser_reader.getImg(1)

        frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
        frame = ser_reader.stretch(frame, 0.3, 0.0001, 0.001, 0.001, 0.001)

        disk_detect = DiskDetector(frame)
        disk_detect.detect()
        (x,y,r) = disk_detect.calculate_disk_coordinates()
        polarbbox = disk_detect.get_polar_bounding_box()
        p_corner1,p_corner2 = polarbbox
        r,a = p_corner1
        x,y = int(r*self.new_r*np.cos(a)), int(r*self.new_r*np.sin(a))
        corner1 = (x+self.center_x,y+self.center_y)
        r,a = p_corner2
        x,y = int(r*self.new_r*np.cos(a)), int(r*self.new_r*np.sin(a))
        corner2 = (x+self.center_x,y+self.center_y)
        cv2.rectangle(image_copy, corner1, corner2,(0, 255,0),cv2.FILLED)
        cv2.rectangle(self.base_image, corner1, corner2, (0, 255,0),2)
        alpha = 0.4
        self.base_image = cv2.addWeighted(image_copy, alpha, self.base_image, 1 - alpha, 0)

        self.window['-IMAGE-'].update(data=self.image_helper.cv2_to_sg(self.base_image))

    def file_call_back(self,path):
        init_size = -1
        ok=False
        if os.access(path, os.R_OK):
            while not ok:
                try:
                    with open(path, 'r') as file:
                        ok = True
                    
                except PermissionError as e:
                    print(f"Erreur de permission : {e}")
                    time.sleep(2)
            
            self.detect_and_display(path)

