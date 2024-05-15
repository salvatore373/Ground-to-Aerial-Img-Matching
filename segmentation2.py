from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import matplotlib.pyplot as plt

def segmentation(device):
    model = SegformerForSemanticSegmentation.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")

    image = Image.open('sat_img.png')

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted = torch.argmax(logits, dim=1)
    predicted = predicted.squeeze().cpu().numpy()   

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Immagine Originale")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted, cmap="jet")
    plt.title("Segmentazione")

    plt.show()

    return predicted

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segmentation(device)


if __name__ == '__main__':
    main()
