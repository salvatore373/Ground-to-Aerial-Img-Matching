from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import matplotlib.pyplot as plt

def segmentation(device, img):
    model = SegformerForSemanticSegmentation.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")

    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted = torch.argmax(logits, dim=1)
    predicted = predicted.squeeze().cpu().numpy()   

    return predicted

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segmentation(device)


if __name__ == '__main__':
    main()
