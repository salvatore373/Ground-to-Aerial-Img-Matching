import torch

from gen_model.feat_extr.vgg import VGG


def test_network():
    rand_img = torch.rand(3, 128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg_model = VGG(device)
    res = vgg_model(rand_img)
    print()


def main():
    test_network()


if __name__ == '__main__':
    main()
