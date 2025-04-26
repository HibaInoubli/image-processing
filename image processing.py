import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Charger et afficher l'image
def load_and_show_image(image_path):
    # Lire l'image en format RGB 
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Image RGB")
    plt.axis('off')
    plt.show()
    return img_rgb

# convertir l'image en niveaux de gris
def convert_to_grayscale(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Image en niveaux de gris")
    plt.axis('off')
    plt.show()
    return img_gray

# afficher les composantes (R, V, B) séparées
def show_rgb_components(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(R, cmap='Reds')
    axs[0].set_title("Composante Rouge")
    axs[0].axis('off')

    axs[1].imshow(G, cmap='Greens')
    axs[1].set_title("Composante Verte")
    axs[1].axis('off')

    axs[2].imshow(B, cmap='Blues')
    axs[2].set_title("Composante Bleue")
    axs[2].axis('off')

    plt.show()

# créer une image BitMap noir & blanc
def create_black_and_white_image():
    black_white_img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    black_white_img_large = np.kron(black_white_img, np.ones((100, 100)))
    plt.imshow(black_white_img_large, cmap='gray')
    plt.title("Image Noir et Blanc")
    plt.axis('off')
    plt.show()
    return black_white_img_large

# Afficher le spectre de Fourier
def show_fourier_spectrum(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Spectre de Fourier")
    plt.axis('off')
    plt.show()

# Fonction principale 
def main(image_path):
    # Charger et afficher l'image RGB
    img_rgb = load_and_show_image(image_path)
    
    # Conversion en niveaux de gris et affichage
    img_gray = convert_to_grayscale(img_rgb)
    
    # Affichage des composantes RGB
    show_rgb_components(img_rgb)
    
    # Création de l'image BitMap et affichage
    black_white_img = create_black_and_white_image()
    
    # Affichage du spectre de Fourier de l'image noir et blanc
    show_fourier_spectrum(black_white_img)

if __name__ == "__main__":
    image_path = 'images/img.jpg'
    main(image_path)
