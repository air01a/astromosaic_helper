import av
import cv2
import numpy as np
from scipy.ndimage import shift
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2


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
    return restored_image

def wiener_deconvolution2(image, kernel, noise_var=1e-3):
    gray_image =  color.rgb2gray(image)
    psf = np.ones((5, 5)) / 30
    rng = np.random.default_rng()
    img = conv2(gray_image, psf, 'same')
    #img += 0.1 * img.std() * rng.standard_normal(img.shape)
    deconvolved  = restoration.richardson_lucy(img, psf, num_iter=30)
    return deconvolved

def stack_sharpest_images(video_path, num_images_to_stack):
    container = av.open(video_path)

    # Vérifier si la vidéo contient des images
    if not container.streams.video:
        raise ValueError("Aucun flux vidéo trouvé dans le fichier.")

    video_stream = container.streams.video[0]

    sharpness_scores = []
    for idx, packet in enumerate(container.demux(video_stream)):
        for frame in packet.decode():

            
            #image = np.array(frame.to_image())

            # Pour les images monochromes, nous prenons simplement un seul canal (par exemple, rouge)
            #image = image[:, :, 0]
            # Convertir le cadre en une image cv2
            image = frame.to_ndarray(format='bgr24')

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

    container.close()
    return stacked_image

# Exemple d'utilisation
video_path = "../2023-08-07-0826_3-U-L-Jup.ser"
num_images_to_stack = 100
kernel_size = 5
kernel_sigma = 1.0
kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
kernel = np.outer(kernel, kernel)


stacked_image = stack_sharpest_images(video_path, num_images_to_stack)
restored_image = wiener_deconvolution2(stacked_image, kernel)

cv2.imshow("Stacked Image", stacked_image)
cv2.imshow("Wiener wavelet",restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()