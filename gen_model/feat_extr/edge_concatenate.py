import cv2
import numpy as np

def get_edgemap(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Verifica che l'immagine in scala di grigi abbia la profondità corretta
    if gray_image.dtype != np.uint8:
        raise ValueError("L'immagine in scala di grigi deve essere di tipo uint8 (8 bit).")
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    edges = edges / 255.0  # Normalize to range [0, 1]
    edges = edges.astype(np.float32)
    return edges

def concatenate_image_and_edgemap(image, edgemap):
    edgemap = np.expand_dims(edgemap, axis=2)  # Add channel dimension
    concatenated = np.concatenate((image, edgemap), axis=2)
    return concatenated