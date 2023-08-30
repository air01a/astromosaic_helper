import cv2
import numpy as np
from PIL import Image
import os

def reconstruct_solar_image(image_paths, output_image_path):
    # Charger les images
    images = [cv2.imread(image_path) for image_path in image_paths]

    # Vérifier que toutes les images ont la même taille
    image_width, image_height = images[0].shape[:2]
   # for image in images:
    #    if image.shape[:2] != (image_width, image_height):
     #       raise ValueError("Toutes les images doivent avoir la même taille.")

    # Calculer la transformation pour assembler les images
    stitcher = cv2.Stitcher_create(mode=1)
   
   
    status, reconstructed_image = stitcher.stitch(images)

    if status == cv2.STITCHER_OK:
        # Sauvegarder l'image reconstituée
        cv2.imwrite(output_image_path, reconstructed_image)
    else:
        print("Impossible de reconstituer le soleil. Veuillez vérifier les images fournies.")
        

# Exemple d'utilisation
folder_path = 'test/'
image_paths = [folder_path+file for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
print(image_paths)
#image_paths = ["test/image1.png", "test/image2.png", "test/image3.png", "test/image4.png", "test/image5.png", "test/image6.png","test/image7.png","test/image8.png","test/image9.png"]
output_image_path = "mosaic_aligned.png"
output_image_path = "soleil_reconstitue.png"
reconstruct_solar_image(image_paths, output_image_path)

