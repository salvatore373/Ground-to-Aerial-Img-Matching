import torch
import torchvision
from torchvision.models import VGG16_Weights
import numpy as np
import matplotlib.pyplot as plt
import torch

from transformation import Transformation
from model.vgg import VGG16

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
    vgg_test(device)

if __name__ == '__main__':
    main()
