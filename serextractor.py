import av
import cv2
import numpy as np

def blur_index(image):
    # Charger l'image en niveaux de gris
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculer le gradient de l'image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculer l'indice de flou comme la somme des carrés des gradients
    blur_index_value = np.sum(gradient_x**2) + np.sum(gradient_y**2)

    return blur_index_value

def extract_image_from_ser(video_path, frame_number, output_image_path):
    container = av.open(video_path)

    # Vérifier si la vidéo contient des images
    if not container.streams.video:
        raise ValueError("Aucun flux vidéo trouvé dans le fichier.")

    video_stream = container.streams.video[0]
    
    # Accéder aux images souhaitées en utilisant le numéro du cadre
    for frame in container.decode(video=0):
        print(blur_index(cv2.cvtColor(frame.to_ndarray(format='bgr24'), cv2.COLOR_BGR2RGB)))

        # Sauvegarder l'image extraite
        frame.to_image().save(output_image_path+str(frame.index)+'.png')
        if frame.index>100:
            break


    container.close()

# Exemple d'utilisation
video_path = "C:/Users/eniquet/Pictures/soleil raw/"
output_image_path = "moon"
frame_number = 100  # Numéro du cadre que vous voulez extraire




extract_image_from_ser(video_path, frame_number, output_image_path)