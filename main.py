import torch

from model.vgg import VGG16


def vgg_test(device):
    inp = torch.randn((3, 224, 224)).to(device)

    model = VGG16(3, device)
    r = model(inp)
    print(r)


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_test(device)


if __name__ == '__main__':
    main()
