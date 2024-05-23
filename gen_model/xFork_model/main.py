import cv2
import numpy as np
import edge_concatenate as ed
import torch
import xFork_gen as g

def main():
    image = cv2.imread('path_to_your_ground_view_image.jpg')
    image = image / 255.0  # Normalize to range [0, 1]
    image = image.astype(np.float32)

    edgemap = ed.get_edgemap(image)

    concatenated_input = ed.concatenate_image_and_edgemap(image, edgemap)

    input_tensor = torch.from_numpy(concatenated_input.transpose((2, 0, 1))).unsqueeze(0)  # Converti a [B, C, H, W]

    # Definisci il modello
    input_nc = 4  # 3 channels for RGB + 1 for edgemap
    output_nc = 3  # Numero di canali per l'immagine di output (es. immagine satellitare)
    output_nc_seg = 1  # Numero di canali per l'output di segmentazione
    ngf = 64  # Numero di filtri nella prima convoluzione

    model = g.xForkGenerator(input_nc, output_nc_seg, output_nc, ngf)
    model = model.cuda()  # Se stai usando una GPU

    # Passa l'input attraverso il modello
    image_out, segmentation_out = model(input_tensor)

    # Converti l'output in numpy array e visualizzalo
    image_out = image_out.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    segmentation_out = segmentation_out.squeeze().cpu().detach().numpy().transpose((1, 2, 0))

    # Visualizza l'immagine generata e la segmentazione
    cv2.imshow('Generated Image', image_out)
    cv2.imshow('Segmentation Output', segmentation_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

