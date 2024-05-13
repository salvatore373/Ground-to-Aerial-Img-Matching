import torch
import torchvision
from torchvision.models import VGG16_Weights
import numpy as np
import matplotlib.pyplot as plt
import torch

from transformation import Transformation
from model.vgg import VGG16

def correlation(device):
    H = 3
    Ws = 5
    Wv = 3
    C = 2

    Fs1 = np.random.rand(H, Ws, C)
    Fg1 = np.random.rand(H, Wv, C)

    Fs2 = np.random.rand(H, Ws, C)
    Fg2 = np.random.rand(H, Wv, C)

    Fs3 = np.random.rand(H, Ws, C)
    Fg3 = np.random.rand(H, Wv, C)

    Fs4 = np.random.rand(H, Ws, C)
    Fg4 = np.random.rand(H, Wv, C)

    pairs = [(Fs1, Fg1), (Fs2, Fg2), (Fs3, Fg3), (Fs4, Fg4)]

    t = Transformation('correlation')
    img_correlation = t.correlation(Fs1, Fg1)
    print("Correlation Scores: ", img_correlation)

    orientation, similiraty = t.generate_similarity_matrix(pairs)
    print("Similarity Scores: ", similiraty)
    print("Orientation: ", orientation)

def polar(device):
    #load the image
    img = plt.imread('sat_img.png')
    
    aerial_size = 1200
    height = 128
    width = 512
    # img = np.random.rand(100, 100)

    t = Transformation('polar', aerial_size, height, width)
    img_polar = t.polar(img)
    print(f"Shape of the original image: {img.shape} and shape of the polar image: {img_polar.shape}")
    
    plt.imshow(img_polar)
    plt.show()
    return img_polar
    
def vgg_test(device):
    inp = torch.randn((3, 224, 224)).to(device)

    model = VGG16(3, include_classifier_part=False, device=device)
    model.load_imagenet_weights_feature_extr()

    r = model(inp)
    print(r.size())


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    polar(device)
    correlation(device)
    vgg_test(device)

if __name__ == '__main__':
    main()
