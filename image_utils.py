import cv2
import PySimpleGUI as sg
import sys
from os import path
from serreader import SerReader
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from fitsutils import open_and_stretch_fits

class ImageHelper:
    def __init__(self):
        self.image_base = None
        self.width, self.height = None, None
        self.center_x, self.center_y = None, None

    @staticmethod
    def is_image_supported_format(image_path):
        file, extension = path.splitext(image_path)
        if extension in ['.ser', '.jpg', '.png', '.tif', '.bmp','.fits']:
            return True
        return False
    
    @staticmethod
    def debayer(image, bayer_pattern):
        cv2_debayer_dict = {

            "BG": cv2.COLOR_BAYER_BG2RGB,
            "GB": cv2.COLOR_BAYER_GB2RGB,
            "RG": cv2.COLOR_BAYER_RG2RGB,
            "GR": cv2.COLOR_BAYER_GR2RGB
        }


        cv_debay = bayer_pattern[3] + bayer_pattern[2]
        try:
            debayered_data = cv2.cvtColor(image, cv2_debayer_dict[cv_debay])
        except KeyError:
            print(f"unsupported bayer pattern : {bayer_pattern}")
        except cv2.error as error:
            print(f"Debayering error : {str(error)}")

        return debayered_data

        
    @staticmethod
    def open_image_file(image_path):
        file, extension = path.splitext(image_path)
        frame = None

        if extension=='.ser' or extension=='.fits':
            if extension=='.ser':
                ser_reader = SerReader(image_path)
                frame = ser_reader.getImg(1)
                frame = ImageHelper.stretch(frame, 0.5, 0.0001, 0.001, 0.001, 0.001)

            else:
                frame = open_and_stretch_fits(image_path).data
            if frame.shape[2]==1:
                frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
#            cv2.imshow('test',frame)
#            cv2.waitKey(0)

            
            
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
        
    @staticmethod
    def stretch(self, image, strength=0.1, alpha0=1,alpha1=1, alpha2=1, alpha3=1):


        min_val = np.percentile(image, strength)
        max_val = np.percentile(image, 100 - strength)
        image = (np.clip((image - min_val) * (65535.0 / (max_val - min_val) ), 0, 65535)/65535).astype(np.double)


        coeffs = pywt.dwt2(image, 'bior1.3')
        cA, (cH, cV, cD) = coeffs
        cA_smoothed = gaussian_filter(cA, sigma=alpha0)
        cH_smoothed = gaussian_filter(cH, sigma=alpha1)
        cV_smoothed = gaussian_filter(cV, sigma=alpha2)
        cD_smoothed = gaussian_filter(cD, sigma=alpha3)
        coeffs_modified = (cA_smoothed, (cH_smoothed, cV_smoothed, cD_smoothed))
        image_sharpened = pywt.idwt2(coeffs_modified, 'bior1.3')

        '''sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharped_img = cv2.filter2D(image_sharpened, -1, sharpen_filter)'''
        #kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)
        #kernel = 1/3 * kernel
        #image_sharpened = cv2.filter2D(image_sharpened*65535, -1, kernel)
        image_sharpened = np.clip(image_sharpened*65535, 0, 65535).astype(np.uint16)
        return image_sharpened
