from stitching import Stitcher
stitcher = Stitcher( detector="brisk",confidence_threshold=0.8)
import os
import cv2

folder_path = 'test/'
image_paths = [folder_path+file for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
print(image_paths)
panorama = stitcher.stitch(image_paths)
cv2.imwrite("test.png", panorama)
