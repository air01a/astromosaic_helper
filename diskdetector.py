import cv2
import numpy as np
import math
import os
import time

class DiskDetector:
    width = 640

    def __init__(self, image):
        h = int(image.shape[0]*self.width/image.shape[1])
        self.image = cv2.resize(image, (self.width, h))
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self.image_blur = cv2.convertScaleAbs(self.image_blur, alpha=5, beta=0)
        self.center_x, self.center_y, self.radius = None, None, None
        self.height = h
        self.diag = np.sqrt(self.width**2+self.height**2)

    def _max(self, contour):
        return cv2.arcLength(contour,False)


    def detect(self):
        edges = cv2.Canny(self.image_blur, threshold1=30, threshold2=70)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.disk = max(contours, key=cv2.contourArea)
        epsilon = 0.0015*cv2.arcLength(self.disk,True)
        self.approx = cv2.approxPolyDP(self.disk,epsilon,True)
        #cv2.drawContours(self.image, contours, -1, (0,0,255), 3)

    def calculate_disk_coordinates(self):
        
        points = self.approx[:, 0, :] 
        num_points = len(points)

        #sum_center_x, sum_center_y = 0,0
        #sum_radius=0
        #n_points=0
        #print(num_points)
        #np.random.shuffle(points)
        #for i in range(2): #range(int(num_points)):
        x1, y1 = points[0]
        x2, y2 = points[int(num_points / 2)]
        x3, y3 = points[num_points-1]

        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        center_x=(((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D)
        center_y=(((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D)

        radius = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
        #n_points +=1
        #    else:
        #        print("D=0")
        #self.center_x = (sum_center_x/(self.width*n_points))
        #self.center_y = (sum_center_y/(self.height*n_points))
        self.center_x = center_x / self.width
        self.center_y = center_y / self.height
        self.radius = radius / self.diag
        #self.radius = (sum_radius/(np.sqrt(self.width**2 + self.height**2)*n_points))

        #print(self.center_x*self.width, self.center_y*self.height, self.radius*(np.sqrt(self.width**2 + self.height**2)))
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
    
    def draw_image_circle(self, image=None):
        if image is None:
            image = self.image.copy()
        if self.center_x!=None:
            cv2.circle(image, (int(self.center_x*image.shape[1]), int(self.center_y*image.shape[0])), int(self.radius*np.sqrt(image.shape[1]**2+image.shape[0]**2)), (0, 255, 0), 2)

        return image
    

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

    folder_path = 'test/'
    image_paths = [folder_path+file for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
    bbox=[]
    image_base = cv2.imread('base.png')
    h = int(image_base.shape[0]*640/image_base.shape[1])
    image_base=cv2.resize(image_base, (640, h))
    (base_w,base_h) = (image_base.shape[1], image_base.shape[0])
    center_x = int(base_w/2)
    center_y = int(base_h/2)


    """dd=DiskDetector(image_base)
    dd.detect()
    dd.calculate_disk_coordinates()
    (x,y,r)  = dd.get_disk_coordinates()
    cv2.imshow('test',dd.draw_image_circle(image_base))
    cv2.waitKey(10)"""
    

 
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
        cv2.waitKey(10)
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



    







