import cv2
import numpy as np
import edge_concatenate as ed
import torch
import xFork_gen as g

def main():
    # Leggi l'immagine dal file
    image = cv2.imread('sat_img.png')

    # Genera la mappa dei bordi
    edgemap = ed.get_edgemap(image)
    
    # Converti la mappa dei bordi da float32 a uint8 per la visualizzazione
    edgemap_display = (edgemap * 255).astype(np.uint8)

    # Converti la mappa dei bordi in 3 canali per la concatenazione
    edgemap_display = cv2.cvtColor(edgemap_display, cv2.COLOR_GRAY2BGR)

    scale_percent = 50 # ridimensiona al 50% della dimensione originale
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Ridimensiona entrambe le immagini
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_edgemap = cv2.resize(edgemap_display, dim, interpolation=cv2.INTER_AREA)
    
    # Concatenare le due immagini orizzontalmente
    concatenated_image = np.hstack((resized_image, resized_edgemap))

    # Visualizza l'immagine concatenata
    cv2.imshow('Image and Edgemap', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



