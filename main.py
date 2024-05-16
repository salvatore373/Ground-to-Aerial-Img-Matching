import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
import numpy as np
import matplotlib.pyplot as plt
import torch

from model.data import CrossViewDataset
from model.trainer import Trainer
from transformation import Transformation
from model.vgg import VGG16

import segmentation
import segmentation2


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


def image_segmentation(device):
    # load the non-segmented image
    img = plt.imread('data/CVUSA_subset/bingmap/input0000008.png')

    # load the segmented image
    seg_img = plt.imread('data\CVUSA_subset\segmap\output0000008.png')

    seg_img1 = segmentation.segmentation(device, img)
    seg_img2 = segmentation2.segmentation(device, img)

    # show the three images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(seg_img)
    axes[0].set_title('The original segmented image')
    axes[0].axis('off')

    axes[1].imshow(seg_img1)
    axes[1].set_title('First segmentation model')
    axes[1].axis('off')

    axes[2].imshow(seg_img2)
    axes[2].set_title('Second segmentation model')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def polar(device):
    # load the image
    img = plt.imread('data/CVUSA_subset/bingmap/input0000008.png')

    # load a polarized image for check the dimension.
    img2 = plt.imread('data/CVUSA_subset/polarmap/normal/input0000008.png')

    aerial_size = img.shape[0]
    height = 128
    width = 512
    # img = np.random.rand(100, 100)

    t = Transformation('polar', aerial_size, height, width)
    img_polar = t.polar(img)
    print(f"Shape of the original image: {img.shape} and shape of the polar image: {img_polar.shape}")

    #plt.imshow(img_polar)
    #plt.show()
    print(f"Shape of the polarized image for comparison: {img2.shape}")

    # show the two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_polar.permute(1, 2, 0))
    axes[0].set_title('My polar image')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title('Original polar image')
    axes[1].axis('off') 

    plt.tight_layout()
    plt.show()

    return img_polar


def vgg_test(device):
    # inp = torch.randn((3, 224, 224)).to(device)
    inp = torch.randn((1, 3, 128, 512)).to(device)

    model = VGG16(3, include_classifier_part=False, circular=True, device=device)
    model.load_imagenet_weights_feature_extr()

    r = model(inp)
    r2 = model(inp)
    q = torch.cat((r, r2), dim=0 if len(r.size()) == 3 else 1)
    print(r.size())


def load_data(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"
    valCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val-19zl.csv"

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

    batch_size = 64
    epochs = 2

    train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device)
    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device)
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Trainer()


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load_data(device)
    # polar(device)
    # correlation(device)
    vgg_test(device)
    #image_segmentation(device)


if __name__ == '__main__':
    main()
