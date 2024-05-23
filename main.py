import os
from pathlib import Path

import torchvision
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from model.data import CrossViewDataset, ImageTypes
from model.san import SAN
from model.trainer import Trainer
from model.transformation import Transformation
from model.vgg import VGG16

from model import segmentation


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
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"
    valCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val-19zl.csv"

    train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device)

    i = 0
    while input('Press enter to display an image: ') != ' ':
        fig, axs = plt.subplots(1, 3)
        img1, img2, img3 = train_dataset.__getitem__(i)
        img1, img2, img3 = np.transpose(img1, (1, 2, 0)), np.transpose(img2, (1, 2, 0)), np.transpose(img3, (1, 2, 0)),
        axs[0].imshow(img1)
        axs[1].imshow(img2)
        axs[2].imshow(img3)
        plt.show()
        i += 1


def train(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"
    valCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val-19zl.csv"

    batch_size = 8
    epochs = 30

    train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                     dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                      ImageTypes.Ground])
    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                           ImageTypes.Ground])
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    san_model = SAN(input_is_transformed=True)
    trainer = Trainer(san_model)
    trainer.train(training_dataloader,
                  validation_dataloader,
                  epochs=epochs,
                  loss_function=san_model.triplet_loss,
                  optimizer=optim.Adam,
                  learning_rate=10e-5, weight_decay=0.01)


def comp_mean_std_dev(path_to_dir, channels=3, width=128, height=128):
    saved_imgs_filenames = [f for f in os.listdir(path_to_dir) if f.endswith('.png') and not f.startswith('._')]
    n = len(saved_imgs_filenames)
    # S1 = torch.zeros(channels)
    # S2 = torch.zeros(channels)
    # for ind, filename in enumerate(saved_imgs_filenames):
    #     img = torchvision.io.read_image(f'{path_to_dir}/{filename}')
    #     curr_mean = img.mean(dim=(1, 2), dtype=torch.float32)
    #
    #     S1 = S1.add(curr_mean)
    #     S2 = S2.add(curr_mean.pow(2))
    #     break
    # # Compute mean and std dev
    # mean = S1.div(n)
    # std = torch.sqrt(S2.div(n) - mean.pow(2))

    runn_mean = torch.zeros(channels)
    M2 = torch.zeros(channels)
    prev_runn_mean = torch.zeros(channels)
    for ind, filename in enumerate(saved_imgs_filenames):
        img = torchvision.io.read_image(f'{path_to_dir}/{filename}')
        curr_mean = img.mean(dim=(1, 2), dtype=torch.float32)

        runn_mean = runn_mean.add((curr_mean - runn_mean).div(n))
        M2 = M2.add((curr_mean - prev_runn_mean) * (curr_mean - runn_mean))

        prev_runn_mean = runn_mean

    # Compute mean and std dev
    mean = runn_mean
    std = M2.div(n)
    return mean, std


def compute_polar_imgs(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    train_csv = f"{dataset_path}/train-19zl.csv"
    valid_csv = f"{dataset_path}/val-19zl.csv"

    output_dir_seg = f"{dataset_path}my_segsat_polar"
    output_dir_sat = f"{dataset_path}my_sat_polar"
    Path(output_dir_seg).mkdir(parents=False, exist_ok=True)
    Path(output_dir_sat).mkdir(parents=False, exist_ok=True)

    train_dataset = CrossViewDataset(train_csv, base_path=dataset_path, device=device,
                                     dataset_content=[ImageTypes.Sat, ImageTypes.SegmentedSat, ImageTypes.Ground])
    validation_dataset = CrossViewDataset(valid_csv, base_path=dataset_path, device=device,
                                          dataset_content=[ImageTypes.Sat, ImageTypes.SegmentedSat, ImageTypes.Ground])
    joint_dataset = ConcatDataset([train_dataset, validation_dataset])

    aerial_size = 370
    height = 128
    width = 512
    t = Transformation('polar', aerial_size, height, width)

    for sat, segm_sat, _, sat_id, segm_sat_id, _ in tqdm(joint_dataset):
        sat_polar = t.polar(sat)
        segm_polar = t.polar(segm_sat)
        torchvision.utils.save_image(sat_polar, f'{output_dir_sat}/{sat_id}.png')
        torchvision.utils.save_image(segm_polar, f'{output_dir_seg}/{segm_sat_id}.png')

    # Load all images
    # sat_pol_filenames = [f for f in os.listdir(output_dir_sat) if f.endswith('.png') and not f.startswith('._')]
    # seg_pol_filenames = [f for f in os.listdir(output_dir_seg) if f.endswith('.png') and not f.startswith('._')]
    # all_sat_tensor = torch.zeros((len(sat_pol_filenames), 3, height, width))
    # all_seg_tensor = torch.zeros((len(seg_pol_filenames), 3, height, width))
    # for ind, filename in enumerate(sat_pol_filenames):
    #     img = torchvision.io.read_image(f'{output_dir_sat}/{filename}')
    #     all_sat_tensor[ind] = img
    # for ind, filename in enumerate(seg_pol_filenames):
    #     img = torchvision.io.read_image(f'{output_dir_seg}/{filename}')
    #     all_seg_tensor[ind] = img
    # # Compute mean and std dev
    # mean_sat = all_sat_tensor.mean(dim=(0, 2, 3))
    # mean_seg = all_seg_tensor.mean(dim=(0, 2, 3))
    # std_sat = all_sat_tensor.std(dim=(0, 2, 3))
    # std_seg = all_seg_tensor.std(dim=(0, 2, 3))
    mean_sat, std_sat = comp_mean_std_dev(output_dir_sat, 3, width, height)
    mean_seg, std_seg = comp_mean_std_dev(output_dir_seg, 3, width, height)
    print('sat polar mean:', mean_sat)
    print('seg polar mean:', mean_seg)
    print('sat polar std:', std_sat)
    print('seg polar std:', std_seg)


def compute_segm_imgs(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    train_csv = f"{dataset_path}/train-19zl.csv"
    valid_csv = f"{dataset_path}/val-19zl.csv"

    output_dir = f"{dataset_path}my_segsat"
    Path(output_dir).mkdir(parents=False, exist_ok=True)

    train_dataset = CrossViewDataset(train_csv, base_path=dataset_path, device=device,
                                     dataset_content=[ImageTypes.Sat, ImageTypes.SegmentedSat, ImageTypes.Ground])
    validation_dataset = CrossViewDataset(valid_csv, base_path=dataset_path, device=device,
                                          dataset_content=[ImageTypes.Sat, ImageTypes.SegmentedSat, ImageTypes.Ground])
    joint_dataset = ConcatDataset([train_dataset, validation_dataset])

    for sat, _, _, sat_id, _, _ in tqdm(joint_dataset):
        sat_channel_last = torch.permute(sat, dims=(2, 1, 0))
        segm_img = segmentation.segmentation(img=sat_channel_last)
        plt.imsave(f'{output_dir}/{sat_id}.png', segm_img)

    # Load all images
    # saved_imgs_filenames = [f for f in os.listdir(output_dir) if f.endswith('.png') and not f.startswith('._')]
    # all_images_tensor = torch.zeros((len(saved_imgs_filenames), 4, 128, 128))
    # for ind, filename in enumerate(saved_imgs_filenames):
    #     img = torchvision.io.read_image(f'{output_dir}/{filename}')
    #     all_images_tensor[ind] = img
    # # Compute mean and std dev
    # mean = all_images_tensor.mean(dim=(0, 2, 3))
    # std = all_images_tensor.std(dim=(0, 2, 3))
    # print('segm mean:', mean)
    # print('segm std:', std)
    mean, std = comp_mean_std_dev(output_dir, 4)
    print('segm mean:', mean)
    print('segm std:', std)


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
    # compute_segm_imgs(device)
    # compute_polar_imgs(device)


if __name__ == '__main__':
    main()
