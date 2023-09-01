import cv2
import PySimpleGUI as sg
import sys
from os import path
from serreader import SerReader

class ImageHelper:
    def __init__(self):
        self.image_base = None
        self.width, self.height = None, None
        self.center_x, self.center_y = None, None

    @staticmethod
    def is_image_supported_format(image_path):
        file, extension = path.splitext(image_path)
        if extension in ['.ser', '.jpg', '.png', '.tif', '.bmp']:
            return True
        return False
    
    @staticmethod
    def open_image_file(image_path):
        file, extension = path.splitext(image_path)
        frame = None

        if extension=='.ser':
            ser_reader = SerReader(image_path)
            frame = ser_reader.getImg(1)
            frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
            frame = ser_reader.stretch(frame, 0.3, 0.0001, 0.001, 0.001, 0.001)

        if extension in ['.jpg','.bmp','.tif','.png']:
            frame = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        
        return frame

    def cv2_to_sg(self, image):

        image_bytes = cv2.imencode('.png', image)[1].tobytes()
        return image_bytes

    def image_resize(self, image, new_width):
        h = int(image.shape[0]*new_width/image.shape[1])
        return cv2.resize(image, (new_width, h))

    def open_image_base(self,type):
        if type==0:
            file = 'base_moon.png'
        else:
            file = 'base_sun.png'

        image_base = cv2.imread(self.get_path(file),cv2.IMREAD_UNCHANGED)
        self.image_base=self.image_resize(image_base,640)
        (base_w,base_h) = (image_base.shape[1], image_base.shape[0])
        self.center_x = int(base_w/2)
        self.center_y = int(base_h/2)
        
        return self.cv2_to_sg(self.image_base)
    
    def min_to_sg_grid(self, thumb):
        (p1,p2,filename, image) = thumb.expand()
        image = self.image_resize(image,200)

        return [[sg.Image(data=self.cv2_to_sg(image[:,:,:3]), size=(200, 100))]]
    
    def get_path(self,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', path.dirname(path.abspath(__file__)))

        return path.join(base_path, relative_path)
        
 
            