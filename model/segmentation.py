import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def segmentation(path_img=None, img=None):
    assert path_img is not None or img is not None

    # Carica il processore dell'immagine
    processor = AutoImageProcessor.from_pretrained("ratnaonline1/segFormer-b4-city-satellite-segmentation-1024x1024")

    # Carica il modello preaddestrato
    model = SegformerForSemanticSegmentation.from_pretrained("ratnaonline1/segFormer-b4-city-satellite-segmentation-1024x1024")

    # Carica un'immagine satellitare da segmentare
    if img is not None:
        input_image = img
    if path_img is not None:
        input_image = Image.open(path_img)

    # Preprocessa l'immagine
    input_image = processor(input_image, return_tensors="pt")

    # Esegui la segmentazione
    with torch.no_grad():
        output = model(**input_image)

    # Ottieni la segmentazione dall'output
    return output.logits.argmax(1).squeeze().cpu().numpy()