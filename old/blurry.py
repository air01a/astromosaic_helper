import cv2
import os

def blur_index(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculer le gradient de l'image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculer l'indice de flou comme la somme des carrés des gradients
    blur_index_value = np.sum(gradient_x**2) + np.sum(gradient_y**2)

    return blur_index_value

def find_most_blurry_image(images_folder):
    # Obtenir la liste de tous les fichiers d'images dans le dossier
    image_files = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Calculer l'indice de flou pour chaque image
    blur_indices = {}
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        blur_indices[image_file] = blur_index(image_path)

    # Sélectionner l'image ayant la valeur de flou la plus élevée
    most_blurry_image = max(blur_indices, key=blur_indices.get)
    most_blurry_index = blur_indices[most_blurry_image]

    return most_blurry_image, most_blurry_index

# Exemple d'utilisation
images_folder = "chemin/vers/votre/dossier_contenant_les_images"
most_blurry_image, blur_index_value = find_most_blurry_image(images_folder)

print(f"L'image la plus floue est : {most_blurry_image}")
print(f"Indice de flou : {blur_index_value}")