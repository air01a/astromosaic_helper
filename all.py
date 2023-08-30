import types
import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.signal import convolve2d as conv2
import os
from stretch import Stretch
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


    def wiener_deconvolution2(image, kernel, noise_var=1e-3):
        gray_image =  color.rgb2gray(image)
        psf = np.ones((5, 5)) / 30
        rng = np.random.default_rng()
        img = conv2(gray_image, psf, 'same')
        #img += 0.1 * img.std() * rng.standard_normal(img.shape)
        deconvolved  = restoration.richardson_lucy(img, psf, num_iter=30)
        return deconvolved
    
    def stretch(self, image, strength=0.1, alpha0=1,alpha1=1, alpha2=1, alpha3=1):


        min_val = np.percentile(image, strength)
        max_val = np.percentile(image, 100 - strength)
        print(min_val,max_val)
        if max_val==min_val:
            return None
        try:
            image = (np.clip((image - min_val) * (65535.0 / (max_val - min_val) ), 0, 65535)/65535).astype(np.double)
        except: 
            return None


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

        """
        coeffs = pywt.dwt2(image, 'haar')
        LL, (LH, HL, HH) = coeffs
        enhanced_LH = LH * 1.2  # Ajustez le facteur selon vos besoins
        enhanced_HL = HL * 1
        enhanced_HH = HH * 1.2
        enhanced_LL = LL * 1
        enhanced_coeffs = enhanced_LL, (enhanced_LH, enhanced_HL, enhanced_HH)
        enhanced_image = pywt.idwt2(enhanced_coeffs, 'haar')
        enhanced_image *= 65535
        image = np.uint16(enhanced_image)

        gamma = 1.03
        adjusted_image = cv2.addWeighted(image, gamma, image, 0, 0)
        # Appliquer un filtre de netteté
        #kernel = np.array([[-1, -1, -1],
        #                [-1,  3, -1],
        #               [-1, -1, -1]])
        #sharpened_image = cv2.filter2D(adjusted_image, -1, kernel)


        return image"""

def calculate_blur_index(image):
    # Convertir l'image en niveaux de gris
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    gray_image = image
    # Calculer le gradient de l'image
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculer l'indice de flou comme la somme des carrés des gradients
    blur_index_value = np.sum(gradient_x**2) + np.sum(gradient_y**2)

    return blur_index_value

def align_images(images):
    # Convertir les images en niveaux de gris
    #gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    gray_images = images

    # Créer un détecteur de points d'intérêt (ORB)
    orb = cv2.ORB_create()

    # Détecter les points d'intérêt et les descripteurs pour chaque image
    keypoints_and_descriptors = [orb.detectAndCompute(gray_image, None) for gray_image in gray_images]

    # Utiliser la première image comme référence pour l'alignement
    reference_image_index = 0
    reference_keypoints, reference_descriptors = keypoints_and_descriptors[reference_image_index]

    # Aligner les autres images sur l'image de référence
    aligned_images = [images[reference_image_index]]
    for i in range(1, len(images)):
        keypoints, descriptors = keypoints_and_descriptors[i]

        # Trouver les correspondances entre les descripteurs de l'image de référence et l'image actuelle
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(reference_descriptors, descriptors)

        # Trier les correspondances en fonction de leur distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Sélectionner les meilleures correspondances (vous pouvez ajuster ce nombre)
        num_good_matches = int(len(matches) * 0.15)
        good_matches = matches[:num_good_matches]

        # Extraire les points correspondants des deux images
        src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calculer la transformation perspective entre les points correspondants
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Appliquer la transformation perspective pour aligner l'image actuelle sur l'image de référence
        aligned_image = cv2.warpPerspective(images[i], M, (images[i].shape[1], images[i].shape[0]))
        aligned_images.append(aligned_image)

    return aligned_images

def wiener_deconvolution(image, kernel, noise_var=1e-3):
    # Appliquer la déconvolution de Wiener
    
    kernel_resized = cv2.resize(kernel, (image.shape[1], image.shape[0]))

    kernel_ft = np.fft.fft2(kernel_resized)
    image_ft = np.fft.fft2(image)

    restored_image_ft = np.conj(kernel_ft) / (np.abs(kernel_ft) ** 2 + noise_var) * image_ft
    restored_image = np.fft.ifft2(restored_image_ft).real
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)


def stack_sharpest_images(video_path, num_images_to_stack):
    ser = SerReader(video_path)
    #container = av.open(video_path)



    sharpness_scores = []
    for n in range(4):
            frame = ser.getImg()

            image=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
            #image = np.array(frame.to_image())

            # Pour les images monochromes, nous prenons simplement un seul canal (par exemple, rouge)
            #image = image[:, :, 0]
            # Convertir le cadre en une image cv2

            image = ser.stretch(image, 0.3, 0.0001, 0.1,0.1,0.1)

            # Calculer l'indice de netteté (indice de flou inversé)
            sharpness_score = -calculate_blur_index(image)

            sharpness_scores.append((sharpness_score, image))

    # Trier les images par niveau de netteté (du plus net au moins net)
    sharpness_scores.sort(reverse=True, key=lambda x: x[0])
    # Extraire les images les plus nettes
    aligned_images = [image for _, image in sharpness_scores[:num_images_to_stack]]

    # Aligner les images
    aligned_images = align_images(aligned_images)

    # Empiler les images les plus nettes
    stacked_image = np.mean(aligned_images, axis=0).astype(np.uint8)


    return stacked_image

# Exemple d'utilisation
video_path = "C:/Users/eniquet/Pictures/soleil raw/17_36_05.ser"
num_images_to_stack = 2
kernel_size = 5
kernel_sigma = 1.0
kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
kernel = np.outer(kernel, kernel)


stacked_image = stack_sharpest_images(video_path, num_images_to_stack)
#restored_image = wiener_deconvolution2(stacked_image, kernel)

cv2.imshow("Stacked Image", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()