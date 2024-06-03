import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from tqdm import tqdm

from san_model.model import segmentation
from san_model.model.data import CrossViewDataset, ImageTypes
from san_model.model.san import SAN
from san_model.model.trainer import Trainer
from san_model.model.transformation import Transformation


def train(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"

    batch_size = 8
    epochs = 30

    full_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                    dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                     ImageTypes.Ground])
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

    # train_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device, normalize_imgs=True,
    #                                  dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
    #                                                   ImageTypes.Ground])
    # validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device, normalize_imgs=True,
    #                                       dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
    #                                                        ImageTypes.Ground])

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(0.1 * len(train_dataset)))
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler)

    san_model = SAN(input_is_transformed=True, device=device)
    trainer = Trainer(san_model, device=device)
    trainer.train(training_dataloader,
                  validation_dataloader,
                  epochs=epochs,
                  loss_function=san_model.triplet_loss,
                  optimizer=optim.Adam,
                  learning_rate=10e-4, weight_decay=0.01)

    import time
    torch.save(san_model.state_dict(),
               f"/Volumes/SALVATORE R/Università/CV/hw_data/model/{int(time.time() * 1000)}.pt")


def evaluate(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    valCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val-19zl.csv"

    batch_size = 8

    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                           ImageTypes.Ground])
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    san_model = SAN(input_is_transformed=True, device=device)
    san_model.load_state_dict(
        torch.load('/Volumes/SALVATORE R/Università/CV/hw_data/saved_models/models_san/1716829223265.pt', map_location=device))
    trainer = Trainer(san_model, device=device)
    print('Starting evaluation...')
    accuracy = trainer.evaluate(validation_dataloader, batch_size, features_output_dim=(16, 4, 64))
    print(f'accuracy: {accuracy:.4f}')


def comp_mean_std_dev(path_to_dir, channels=3):
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
        img = torchvision.io.read_image(f'{path_to_dir}/{filename}').div(255)
        curr_mean = img.mean(dim=(1, 2), dtype=torch.float32)

        runn_mean = runn_mean.add((curr_mean - runn_mean).div(n))
        M2 = M2.add((curr_mean - prev_runn_mean) * (curr_mean - runn_mean))

        prev_runn_mean = runn_mean

    # Compute mean and std dev
    mean = runn_mean
    std = M2.div(n).sqrt()
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

    # Compute mean and std dev
    mean_sat, std_sat = comp_mean_std_dev(output_dir_sat, 3)
    mean_seg, std_seg = comp_mean_std_dev(output_dir_seg, 3)
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

    # Compute mean and std dev
    mean, std = comp_mean_std_dev(output_dir, 4)
    print('segm mean:', mean)
    print('segm std:', std)


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('salvatore')

    # load_data(device)
    # img_polar2 = polar_tensor(device)
    # print("Shape of the polarized image: ", img_polar2.shape)
    # polar(device)
    # correlation(device)
    # vgg_test(device)
    # image_segmentation(device)

    # train(device)
    evaluate(device)
    # compute_segm_imgs(device)
    # compute_polar_imgs(device)


if __name__ == '__main__':
    main()
