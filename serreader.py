# -*- coding: utf-8 -*-

import types
import numpy as np
import cv2

import os
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d as conv2

class SerReader(object):
    def __init__(self,fname):
        self.fname=fname
        self.header=types.SimpleNamespace()
        with open(self.fname,"rb") as f:
            self.header.fileID=f.read(14).decode()
            self.header.luID=int.from_bytes(f.read(4), byteorder='little')
            self.header.colorID=int.from_bytes(f.read(4), byteorder='little')
            """
            Content:
                MONO= 0
                BAYER_RGGB= 8
                BAYER_GRBG= 9
                BAYER_GBRG= 10
                BAYER_BGGR= 11
                BAYER_CYYM= 16
                BAYER_YCMY= 17
                BAYER_YMCY= 18
                BAYER_MYYC= 19
                RGB= 100
                BGR= 101
            """
            if self.header.colorID <99:
                self.header.numPlanes = 1
            else:
                self.header.numPlanes = 3
                
            self.header.littleEndian=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageWidth=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageHeight=int.from_bytes(f.read(4), byteorder='little')
            self.header.PixelDepthPerPlane=int.from_bytes(f.read(4), byteorder='little')
            if self.header.PixelDepthPerPlane == 8:
                self.dtype = np.uint8
            elif self.header.PixelDepthPerPlane == 16:
                self.dtype = np.uint16
            self.header.frameCount=int.from_bytes(f.read(4), byteorder='little')
            self.header.observer=f.read(40).decode()
            self.header.instrument=f.read(40).decode()
            self.header.telescope=f.read(40).decode()
            self.header.dateTime=int.from_bytes(f.read(8), byteorder='little')
            self.imgSizeBytes = int(self.header.imageHeight*self.header.imageWidth*self.header.PixelDepthPerPlane*self.header.numPlanes/8)
            self.imgNum=0
        
    def getImg(self,imgNum=None):
        if imgNum is None:
            pass
        else:
            self.imgNum=imgNum
            
        with open(self.fname,"rb") as f:
            f.seek(int(178+self.imgNum*(self.imgSizeBytes)))
            frame = np.frombuffer(f.read(self.imgSizeBytes),dtype=self.dtype)
            
        self.imgNum+=1
        
        frame = np.reshape(frame,(self.header.imageHeight,self.header.imageWidth,self.header.numPlanes))
        return frame


    def blur_index(self, image):
        # Charger l'image en niveaux de gris
        #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Calculer le gradient de l'image
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculer l'indice de flou comme la somme des carrés des gradients
        blur_index_value = np.sum(gradient_x**2) + np.sum(gradient_y**2)

        return blur_index_value
    

    def unsharp_mask(self,image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 65535 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint16)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened


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

    
max_frames = 1

if __name__ == '__main__':
    folder_path=r'C:/Users/eniquet/Pictures/soleil raw/'
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.ser'))]
    t_alpha0 = [0.0001]
    t_alpha1 = [0.001]
    t_alpha2 = [0.001]
    t_alpha3 = [0.001]
    i = 0
    for file in image_files:
        fname=folder_path+file
        print("%s = %i" % (fname, i))
        ser = SerReader(fname)
        for n in range(min(max_frames,ser.header.frameCount)):
            frame = ser.getImg()
            

            #print(ser.blur_index(frame))
            frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)

            #cv2.imshow('x',frame)

        frame_save = frame.copy()
        for alpha1 in t_alpha1:
            for alpha2 in t_alpha2:
                for alpha3 in t_alpha3:
                    for alpha0 in t_alpha0:
                        frame = frame_save.copy()
                        frame = ser.stretch(frame, 0.3, alpha0, alpha1, alpha2, alpha3)
                        sharpness = ser.blur_index(frame)
                        mean_brightness = np.mean(frame)
                        print("sunbior%f.%f.%f.%f.%i.png;%f" % (alpha0, alpha1, alpha2, alpha3, i,sharpness))
                        #frame = ser.unsharp_mask(frame)
        #frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
                        cv2.imwrite("test/sunbior%f.%f.%f.%f.%i.tif" %(alpha0, alpha1, alpha2, alpha3, i), frame)
                        image = cv2.imread("test/sunbior%f.%f.%f.%f.%i.tif" %(alpha0, alpha1, alpha2, alpha3, i))
                        cv2.imwrite("test/sun%i.bmp" %i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                        i+=1
