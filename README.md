# astromosaic_helper
Watch a directory for SER file, extract 1 image from ser and try to assemblate the mosaic. This program has been written for the Jonckheere Association, in order to facilitate our work on sun and moon observation. 
The goal is to be sure that the entire object has been covered when doing mosaic (sun / moon or big deep sky object)

![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/interface.png?raw=true "sun")

It uses cv2 to adapt image and detect contours, a bit of mathematics to find circle center and radius (find a circle that pass through the different points of a triangle, polar coordinates to change reference, ...), pysimplegui for the interface, and a custom script to read ser files. 

The interface is very simple, simply choose a directory to watch, and if you want to observe the moon or the sun (the image processing will be slightly different as it's a bit more complex to find the contour of the moon due to the presence of many craters and the terminator). The picture displaying the results will be change according to your observation. Just click on "analyse" and put your pictures/video in the correct folder.



# Example with image

![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image1.png?raw=true "sun")   ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image2.png?raw=true "sun")   ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image3.png?raw=true "sun")

![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image4.png?raw=true "sun")   ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image5.png?raw=true "sun")   ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image6.png?raw=true "sun")

![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image7.png?raw=true "sun")  ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image8.png?raw=true "sun")   ![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/image9.png?raw=true "sun")


Result


![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/sun.png?raw=true "sun final") 


# Low contrast sun image

With the Sun, the different algorithm that can be used to stitch the mosaic lack of details to find common points between image. I tried many things with no success. 

So, I have chosen a different approch. I use cv2 to find the contour of the sun (an arc for each image), calculate the center and radius of the arc, then interpolate the picture coordinates to find the correct position of the image in a sun picture.

![Alt text](https://github.com/air01a/astromosaic_helper/blob/main/doc/diskdetector.jpg?raw=true "sun final") 

That was not so easy, but it works :)
