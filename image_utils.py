import cv2

class ImageHelper:
    def __init__(self):
        self.image_base = None
        self.width, self.height = None, None
        self.center_x, self.center_y = None, None

    def cv2_to_sg(self, image):
        image_bytes = cv2.imencode('.png', self.image_base)[1].tobytes()
        return image_bytes

    def open_image_base(self):
        image_base = cv2.imread('base.png')
        h = int(image_base.shape[0]*640/image_base.shape[1])
        self.image_base=cv2.resize(image_base, (640, h))
        (base_w,base_h) = (image_base.shape[1], image_base.shape[0])
        self.center_x = int(base_w/2)
        self.center_y = int(base_h/2)
        
        return self.cv2_to_sg(self.image_base)