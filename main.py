import torch
import torchvision
from torchvision.models import VGG16_Weights

from model.vgg import VGG16


def vgg_test(device):
    inp = torch.randn((3, 224, 224)).to(device)

    model = VGG16(3, include_classifier_part=False, device=device)
    model.load_imagenet_weights_feature_extr()

    r = model(inp)
    print(r.size())


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_test(device)

if __name__ == '__main__':
    main()
