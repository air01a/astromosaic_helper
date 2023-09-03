import cv2
import numpy as np
import math
import os
#import time
#import matplotlib.pyplot as plt


CANNY_THRESHOLD1=30
CANNY_THRESHOLD2=70
IMAGE_WIDTH=640
BINARY_THRESHOLD=40
EPSILON=0.0015
MEAN_FACTOR=0.5
SUR_EXPO_FACTOR=1.5

class DiskDetector:
    width = IMAGE_WIDTH

    def __init__(self, image,alpha=SUR_EXPO_FACTOR):
        # Resize to 640x?
        h = int(image.shape[0]*self.width/image.shape[1])
        self.image = cv2.resize(image, (self.width, h))

        # Convert to 8 bits
        if self.image.dtype==np.uint16:
            self.image = (self.image/256).astype(np.uint8)

        # Blur to have soft contour
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Binarize image to avoid problems with moon craters
        histo = cv2.calcHist(gray, [0], None, [256], [0, 256])
        peak = np.argmax(histo)
        peak = 1
        #plt.figure()
        #plt.title("Histogramme de l'image")
        #plt.xlabel("Valeur des pixels")
        #plt.ylabel("Fréquence")
        #plt.plot(histo)
        #plt.xlim([0, 256])
        #plt.show()
        _, self.image_blur = cv2.threshold(gray, BINARY_THRESHOLD*peak, 255, cv2.THRESH_BINARY)  
        self.center_x, self.center_y, self.radius = None, None, None
        self.height = h
        self.diag = np.sqrt(self.width**2+self.height**2)


    def _max(self, contour):

        return cv2.arcLength(contour, True)

    
    def detect(self):

        # Apply Canny filter
        edges = cv2.Canny(self.image_blur, threshold1=CANNY_THRESHOLD1, threshold2=CANNY_THRESHOLD2)

        # Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get contour with max arc length
        self.disk = max(contours, key=self._max)

        # Approximate with polynome
        epsilon = EPSILON*cv2.arcLength(self.disk,True)
        self.approx = cv2.approxPolyDP(self.disk,epsilon,True)
        cv2.drawContours(self.image, [self.approx], -1, (0,0,255), 3)
 



    def calculate_disk_from_points(self, x1,y1, x2, y2, x3,y3):

        # Calculate circle coordinates that pass through three points
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if D!=0:
            center_x=(((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D)
            center_y=(((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D)

            radius = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
            return (center_x, center_y, radius)
        return (None, None, None)


    def calculate_disk_coordinates(self):
        
        points = self.approx[:, 0, :] 
        num_points = len(points)
        #x1, y1 = points[int(num_points/4)]
        #x2, y2 = points[int(num_points / 2)]
        #x3, y3 = points[int(num_points*3/4)]
        
        ''' For all points, calculates the circle (with point(i), point(i+1), point(i+2)'''
        results = []
        for i in range(len(points)-2):
            #x1, y1 = points[0]
            #x2, y2 = points[int(num_points / 2)]
            #x3, y3 = points[-1]
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            x3, y3 = points[i+2]
            cx, cy, cr = self.calculate_disk_from_points(x1,y1,x2,y2,x3,y3)
            if cr!=None:
                results.append(cr)

        ''' Eliminate all points that are far from the mean radius
            It is mandatory as contour can include parts of terminator with moon picture'''

        results=np.array(results)
        mean = np.mean(results)
        sigma = np.std(results)
        points_to_keep = []
        for indice, result in enumerate(results):
            if result>MEAN_FACTOR*mean:
                points_to_keep.append(points[indice])

        ''' We're now good to calculate disk coordinates with extrem points'''

        x1, y1 = points_to_keep[0]
        x2, y2 = points_to_keep[int(len(points_to_keep) / 2)]
        x3, y3 = points_to_keep[-1]
        center_x, center_y, radius = self.calculate_disk_from_points(x1,y1,x2,y2,x3,y3)

        self.center_x = center_x / self.width
        self.center_y = center_y / self.height
        self.radius = radius / self.diag

        return (self.center_x, self.center_y, self.radius)

    def get_disk_coordinates(self):
        return (self.center_x, self.center_y, self.radius)

    def get_arc_length(self):
        arc_length = cv2.arcLength(self.approx, True)

        return arc_length
    
    def draw_image_contour(self, image=None):
        if image is None:
            image = self.image.copy()
        cv2.drawContours(image, [self.approx], -1, (0,0,255), 3)
        return image
    
    def draw_image_circle(self, image=None, thickness=2):
        if image is None:
            image = self.image.copy()
        if self.center_x!=None:
            cv2.circle(image, (int(self.center_x*image.shape[1]), int(self.center_y*image.shape[0])), int(self.radius*np.sqrt(image.shape[1]**2+image.shape[0]**2)), (0, 255, 0), thickness)

        return image
    
    # Calculate polar coordinates of the picture (from the disk center)
    def get_polar_bounding_box(self):
        coords = [(0,0),(self.width,self.height)]
        result=[]
        for c in coords:
            x,y = c
            x-=self.center_x*self.width
            y-=self.center_y*self.height

            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y,x)
  
            r = r/(self.radius*self.diag)
            result.append((r,theta))
        return result

        



if __name__ == '__main__':
    new_r = 309.83

    folder_path = 'work/'
    image_paths = [folder_path+file for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
    bbox=[]
    image_base = cv2.imread('base_sun.png')
    h = int(image_base.shape[0]*640/image_base.shape[1])
    image_base=cv2.resize(image_base, (640, h))
    (base_w,base_h) = (image_base.shape[1], image_base.shape[0])
    center_x = int(base_w/2)
    center_y = int(base_h/2)

    

 
    for path in image_paths:
        image = cv2.imread(path)
        (w,h) = (image.shape[1], image.shape[0])
        disk_detect = DiskDetector(image)
        disk_detect.detect()
        (x,y,r) = disk_detect.calculate_disk_coordinates()
        arc_length = disk_detect.get_arc_length()
        image = disk_detect.draw_image_circle()
        #disk_detect.draw_image_contour(image)
        cv2.imshow('None approximation' + path, image)
        cv2.waitKey(4000)
        polarbbox = disk_detect.get_polar_bounding_box()
        bbox.append(polarbbox)

    i=0
    colors = [ (0, 255,0)]
    image_copy = image_base.copy()
    for coord in bbox:
        p_corner1,p_corner2 = coord
        r,a = p_corner1
        x,y = int(r*new_r*np.cos(a)), int(r*new_r*np.sin(a))
        corner1 = (x+center_x,y+center_y)
        r,a = p_corner2
        x,y = int(r*new_r*np.cos(a)), int(r*new_r*np.sin(a))
        corner2 = (x+center_x,y+center_y)
        imaimage_basee = cv2.rectangle(image_copy, corner1, corner2, colors[i%len(colors)],cv2.FILLED)
        cv2.rectangle(image_base, corner1, corner2, colors[i%len(colors)],2)
        i+=1

    alpha = 0.4
    image_new = cv2.addWeighted(image_copy, alpha, image_base, 1 - alpha, 0)
    cv2.imshow('Image avec Rectangle', image_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    







