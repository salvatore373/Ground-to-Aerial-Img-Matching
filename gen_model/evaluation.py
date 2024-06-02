import torch
import sys
import os

# Aggiungi la directory superiore al percorso di ricerca dei moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_model.feat_extr.feature_extractor import JointFeatureLearningNetwork, FeatureExtractor
from gen_model.feat_extr.vgg import VGG
import tensorflow as tf
from san_model.model.data import CrossViewDataset, ImageTypes
from torch.utils.data import DataLoader,SequentialSampler, Subset

import numpy as np

def create_dataset(device, dataset_path, validation_path):

    batch_size = 8

    validation_dataset = CrossViewDataset(validation_path, base_path=dataset_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.Ground, ImageTypes.Sat,
                                                           ImageTypes.SyntheticSat])
    
    # Use only the first 503 items
    indices = list(range(503))
    validation_subset = Subset(validation_dataset, indices)

    valid_sampler = SequentialSampler(validation_subset)
    validation_dataloader = DataLoader(validation_subset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    print("Validation dataset created.")
    
    # Visiting the dataset, DEBUG
    query = validation_subset[0]
    print("Query: ", query)
    print("Query type: ", type(query))
    print("Query shape: ", query[0].shape)
    
    return validation_dataloader

def evaluate(device, dataloader):
    # Load the model
    model = FeatureExtractor(device)

    ground_features_jfl = []
    satellite_features_jfl = []
    ground_features_fusion = []
    satellite_features_fusion = []
    ground_truth_indices = []

    # Evaluate the model
    for i, (ground, sat, synt, _, _, _) in enumerate(dataloader):
        ground = ground.to(device)
        sat = sat.to(device)
        synt = synt.to(device)

        with torch.no_grad():
            out_net1, out_net2 = model(ground, sat, synt)
            
            # Joint Feature Learning
            ground_feats_jfl = out_net1[0].cpu().numpy()
            satellite_feats_jfl = out_net1[1].cpu().numpy()

            # Feature Fusion
            fusion_feats = out_net2[1].cpu().numpy()
            fusion_sat_feats = out_net2[0].cpu().numpy()

            ground_features_jfl.append(ground_feats_jfl)
            satellite_features_jfl.append(satellite_feats_jfl)
            ground_features_fusion.append(fusion_feats)
            satellite_features_fusion.append(fusion_sat_feats)
            ground_truth_indices.extend(range(i * len(ground), i * len(ground) + len(ground)))

    # Convert to numpy arrays
    ground_features_jfl = np.vstack(ground_features_jfl)
    satellite_features_jfl = np.vstack(satellite_features_jfl)
    ground_features_fusion = np.vstack(ground_features_fusion)
    satellite_features_fusion = np.vstack(satellite_features_fusion)

    # Calculate distances for Joint Feature Learning
    distances_jfl = np.linalg.norm(ground_features_jfl[:, np.newaxis] - satellite_features_jfl, axis=2)

    # Calculate distances for Feature Fusion
    distances_fusion = np.linalg.norm(ground_features_fusion[:, np.newaxis] - satellite_features_fusion, axis=2)

    # Calculate top-k recalls for Joint Feature Learning
    top_1_recall_jfl = calculate_recall(distances_jfl, ground_truth_indices, 1)
    top_10_recall_jfl = calculate_recall(distances_jfl, ground_truth_indices, 10)
    top_1_percent_recall_jfl = calculate_recall(distances_jfl, ground_truth_indices, int(0.01 * distances_jfl.shape[1]))

    # Calculate top-k recalls for Feature Fusion
    top_1_recall_fusion = calculate_recall(distances_fusion, ground_truth_indices, 1)
    top_10_recall_fusion = calculate_recall(distances_fusion, ground_truth_indices, 10)
    top_1_percent_recall_fusion = calculate_recall(distances_fusion, ground_truth_indices, int(0.01 * distances_fusion.shape[1]))

    print(f'Joint Feature Learning - Top-1 Recall: {top_1_recall_jfl:.4f}')
    print(f'Joint Feature Learning - Top-10 Recall: {top_10_recall_jfl:.4f}')
    print(f'Joint Feature Learning - Top-1% Recall: {top_1_percent_recall_jfl:.4f}')

    print(f'Feature Fusion - Top-1 Recall: {top_1_recall_fusion:.4f}')
    print(f'Feature Fusion - Top-10 Recall: {top_10_recall_fusion:.4f}')
    print(f'Feature Fusion - Top-1% Recall: {top_1_percent_recall_fusion:.4f}')

def calculate_recall(distances, ground_truth_indices, top_k):
    """
    Calculate the recall for top-k matches.
    
    distances: 2D array where distances[i, j] is the distance between ground image i and aerial image j
    ground_truth_indices: 1D array where ground_truth_indices[i] is the index of the correct match for ground image i
    top_k: the value of k for top-k recall
    
    Returns: recall value
    """
    num_queries = distances.shape[0]
    correct_matches = 0
    
    for i in range(num_queries):
        sorted_indices = np.argsort(distances[i])
        if ground_truth_indices[i] in sorted_indices[:top_k]:
            correct_matches += 1
    
    recall = correct_matches / num_queries
    return recall


def main():
    #test_network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CHANGE THIS FOR RUNNING THE CODE
    dataset_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset"
    valCSV = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/val-19zl.csv"

    validation_dataloader = create_dataset(device, dataset_path, valCSV)
    evaluate(device, validation_dataloader)
    # eight_layer_conv_multiscale()
    # three_stream_joint_feat_learning()

if __name__ == '__main__':
    main()
