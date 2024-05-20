import pandas as pd
import torch
import torchvision
from tensorflow.python.keras.layers import Conv2D
from torch import optim
from torch.utils.data import DataLoader
import torchvision as vision
from torchvision.models import VGG16_Weights
import numpy as np
import matplotlib.pyplot as plt
import torch

from model.data import CrossViewDataset
from model.san import SAN
from model.trainer import Trainer
from transformation import Transformation
from model.vgg import VGG16

import segmentation


def image_segmentation(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"

    # load the non-segmented image
    img_path = f'{dataset_path}/bingmap/input0014917.png'
    orig_segm = f'{dataset_path}/segmap/output0014917.png'

    pred = segmentation.segmentation(img_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(pred)
    axes[0].set_title('My segmentation')
    axes[0].axis('off')

    axes[1].imshow(plt.imread(img_path))
    axes[1].set_title('Original image')
    axes[1].axis('off')
    axes[1].axis('off')

    axes[2].imshow(plt.imread(orig_segm))
    axes[2].set_title('Original segm image')
    axes[2].axis('off')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def polar(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"

    # load the image
    img = plt.imread(f'{dataset_path}/bingmap/input0000008.png')

    # load a polarized image for check the dimension.
    img2 = plt.imread(f'{dataset_path}/polarmap/normal/input0000008.png')

    # Permute the dimension of the image
    img = np.transpose(img, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))

    aerial_size = 370
    height = 128
    width = 512
    # img = np.random.rand(100, 100)

    t = Transformation('polar', aerial_size, height, width)
    img_polar = t.polar(img)
    print(f"Shape of the original image: {img.shape} and shape of the polar image: {img_polar.shape}")

    # plt.imshow(img_polar)
    # plt.show()
    print(f"Shape of the polarized image for comparison: {img2.shape}")

    # show the two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_polar.permute(1, 2, 0))
    axes[0].set_title('My polar image')
    axes[0].axis('off')

    axes[1].imshow(np.transpose(img2, (1, 2, 0)))
    axes[1].set_title('Original polar image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return img_polar


def polar_tensor(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    # load the image
    img = plt.imread(f'{dataset_path}/bingmap/input0000008.png')

    # load a polarized image for check the dimension.
    img2 = plt.imread(f'{dataset_path}/polarmap/normal/input0000008.png')

    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1)

    aerial_size = 370
    height = 128
    width = 512

    t = Transformation('polar', aerial_size, height, width)
    img_polar = t.polar(img_tensor)
    print(f"Shape of the original image: {img2_tensor.shape} and shape of the polar image: {img_polar.shape}")

    # plt.imshow(img_polar)
    # plt.show()
    print(f"Shape of the polarized image for comparison: {img2_tensor.shape}")

    # show the two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_polar.permute(1, 2, 0))
    axes[0].set_title('My polar image')
    axes[0].axis('off')

    axes[1].imshow(img2_tensor.permute(1, 2, 0))
    axes[1].set_title('Original polar image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return img_polar


def vgg_test(device):
    inp = torch.randn((3, 224, 224)).to(device)

    model = VGG16(3, include_classifier_part=False, device=device)
    model.load_imagenet_weights_feature_extr()

    r = model(inp)
    print(r.size())


def load_data(device):
    dataset_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset"
    trainCSV = "data/CVUSA_subset/train-19zl.csv"
    valCSV = "data/CVUSA_subset/val-19zl.csv"

    # id_list = []
    # id_idx_list = []
    # with open(trainCSV, 'r') as file:
    #     idx = 0
    #     for line in file:
    #         data = line.split(',')
    #         pano_id = (data[0].split('/')[-1]).split('.')[0]
    #         # satellite filename, streetview filename, pano_id
    #         id_list.append([data[0].replace('bing', 'polar').replace('jpg', 'png'), data[0], data[1], pano_id])
    #         # polarmap, bingmap, streetview, id (inputXXX)
    #         id_idx_list.append(idx)
    #         idx += 1

    # methods return  batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien

    train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device)
    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device)

    batch_size = 128
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #  Display some samples
    num_samples = 2
    fig, axs = plt.subplots(num_samples, 3)
    for i in range(num_samples):
        img1, img2, img3 = train_dataset.__getitem__(i)
        img1, img2, img3 = np.transpose(img1, (1, 2, 0)), np.transpose(img2, (1, 2, 0)), np.transpose(img3, (1, 2, 0)),
        axs[i][0].imshow(img1)
        axs[i][1].imshow(img2)
        axs[i][2].imshow(img3)

    plt.show()


def train(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"
    valCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val-19zl.csv"

    batch_size = 4
    epochs = 2

    train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device)
    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device)
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    polarmap, bingmap, streetview = train_dataset[0]
    segimgs = vision.io.read_image(  # TODO: replace
        "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/segmap/output0044192.png")
    # plt.imshow(np.transpose(segimgs, (1, 2, 0)))
    # plt.show()
    san_model = SAN(input_is_transformed=False)
    # san_model(streetview.unsqueeze(dim=0), bingmap.unsqueeze(dim=0), segimgs.unsqueeze(dim=0))

    trainer = Trainer(san_model)
    trainer.train(training_dataloader,
                  validation_dataloader,
                  epochs=epochs,
                  loss_function=san_model.triplet_loss,
                  optimizer=optim.Adam,
                  learning_rate=10e-5, weight_decay=0.01)


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load_data(device)
    # img_polar2 = polar_tensor(device)
    # print("Shape of the polarized image: ", img_polar2.shape)
    # polar(device)
    # correlation(device)
    # vgg_test(device)
    # image_segmentation(device)
    train(device)


if __name__ == '__main__':
    main()
