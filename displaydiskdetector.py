import cv2
from serreader import SerReader
from diskdetector import DiskDetector
import numpy as np
import time
import os
import threading
import queue
from image_utils import ImageHelper


class Thumbnail:
    
    def __init__(self, coord1, coord2, filename, image):
        self.coord1 = coord1
        self.coord2 = coord2

        self.filename = filename
        self.image = image



    def expand(self):
        return (self.coord1, self.coord2, self.filename, self.image)


class DisplayDiskDetector():
    def __init__(self, window, image_helper, base_image, new_r):
        self.new_r = new_r
        self.base_image = base_image
        self.window = window
        self.image_helper = image_helper
        self.thumbnails = []

        (base_w,base_h) = (self.base_image.shape[1], self.base_image.shape[0])
        self.center_x = int(base_w/2)
        self.center_y = int(base_h/2)

        self.end = False
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.detector, daemon=True)
        self.thread.start()


    def stop(self):
        self.end=True
        self.queue.put('')


    def detector(self):

        while not self.end:
            item = self.queue.get()
            if item!='':
                self.window.write_event_value('-NEW TASK-',item)
                self._detect_and_display(item)
                self.queue.task_done()
            

    def detect_and_display(self, path):
        self.queue.put(path)

    def _detect_and_display(self, path):
        image_copy = self.base_image.copy()
        frame = ImageHelper.open_image_file(path)


        # Detect disk
        disk_detect = DiskDetector(frame)
        disk_detect.detect()
        (x,y,r) = disk_detect.calculate_disk_coordinates()
        image = disk_detect.draw_image_circle(thickness=10)

        polarbbox = disk_detect.get_polar_bounding_box()


        # Calculate coordinates
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
        
        index = len(self.thumbnails)
        self.thumbnails.append(Thumbnail(corner1,corner2,path,image))

        self.window.write_event_value('-UPDATE MINIATURE-',index)

    def file_call_back(self,path):
        init_size = -1
        ok=False
        if not ImageHelper.is_image_supported_format(path):
            return
        
        if os.access(path, os.R_OK):
            while not ok:
                try:
                    with open(path, 'r') as file:
                        ok = True
                    
                except PermissionError as e:
                    #print(f"Erreur de permission : {e}")
                    time.sleep(2)
            
            self.detect_and_display(path)

