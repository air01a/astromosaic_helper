import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class FullscreenImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visionneuse d'images en plein écran")

        # Créer un canevas pour afficher l'image
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Demander à l'utilisateur de choisir un fichier image
        self.file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if not self.file_path:
            self.root.destroy()
            return

        # Afficher l'image en plein écran
        self.image = Image.open(self.file_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Configurer la fenêtre pour le plein écran
        #self.root.attributes('-fullscreen', True)

        # Lier la touche Echap pour quitter le plein écran
        self.root.bind("<Escape>", self.exit_fullscreen)

    def exit_fullscreen(self, event):
        # Quand la touche Echap est pressée, quitter le plein écran et fermer la fenêtre
        self.root.attributes('-fullscreen', False)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    viewer = FullscreenImageViewer(root)
    root.mainloop()