import torch
import sys
import os

# Aggiungi la directory superiore al percorso di ricerca dei moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_model.feat_extr.feature_extractor import JointFeatureLearningNetwork, FeatureExtractor
from gen_model.feat_extr.vgg import VGG
import tensorflow as tf
from san_model.model.data import CrossViewDataset, ImageTypes
from torch.utils.data import DataLoader,RandomSampler

import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

def create_dataset(device):
    dataset_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset"
    valCSV = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/val-19zl.csv"

    batch_size = 8

    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                           ImageTypes.Ground])
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    print("Validation dataset created.")
    
    # Visiting the dataset
    query = validation_dataset[0]
    print("Query: ", query)
    print("Query type: ", type(query))
    print("Query shape: ", query[0].shape)

      # Accessing the images through the dataloader
    for polar_sat, polar_segm_sat, ground, _, _, _ in validation_dataloader:
        print("Polar satellite image shape: ", polar_sat.shape)
        print("Polar segmented satellite image shape: ", polar_segm_sat.shape)
        print("Ground image shape: ", ground.shape)
        break
    
    return validation_dataloader

def evaluate(device):
    # Load the model
    model = FeatureExtractor(device)

    # Load the dataset
    dataset = create_dataset(device) # TODO: Change the dataset with the correct images

    # Evaluate the model
    for polar_sat, polar_segm_sat, ground, _, _, _ in dataset: # TODO: we must have ground, satellite and synthetic satellite 
        ground = ground.to(device)
        satellite = polar_sat.to(device)
        synthetic_satellite = polar_segm_sat.to(device)

        with torch.no_grad():
            out_net1, out_net2 = model(ground, satellite, synthetic_satellite)

            print("Output of the Joint Feature Learning network: ", out_net1)
            print("Output of the Feature Fusion network: ", out_net2)

def main():
    #test_network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = create_dataset(device)
    #evaluate(device)
    # eight_layer_conv_multiscale()
    # three_stream_joint_feat_learning()

if __name__ == '__main__':
    main()