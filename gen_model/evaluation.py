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
                                          dataset_content=[ImageTypes.Ground, ImageTypes.Sat, ImageTypes.SyntheticSat])
    
    indices = list(range(503))
    validation_subset = Subset(validation_dataset, indices)

    valid_sampler = SequentialSampler(validation_subset)
    validation_dataloader = DataLoader(validation_subset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    print("Validation dataset created.")

    for i, (ground, sat, synt, _, _, _) in enumerate(validation_dataloader):
        print(len(ground))
        print(len(sat))
        print(len(synt))
        break
    
    return validation_dataloader

def evaluate(device, dataloader):
    print("Evaluating the model...")
    print("Loading the model and the weights...")

    gen_model = FeatureExtractor(device)
    gen_model.load_weights(
        'gen_model/feat_extr/gen_weights/jfl_1717343102687.pt',
        'gen_model/feat_extr/gen_weights/ff_1717343102687.pt'
    )

    ground_features_jfl = []
    satellite_features_jfl = []
    synthetic_satellite_features_jfl = []
    ground_features_fe = []
    satellite_features_fe = []
    ground_truth_indices = []

    for i, (ground, sat, synt, _, _, _) in enumerate(dataloader):
        ground = ground.to(device)
        sat = sat.to(device)
        synt = synt.to(device)

        with torch.no_grad():
            out_net1, out_net2 = gen_model(ground, sat, synt)

            # Estrazione delle features per il metodo JFL
            ground_feats_jfl = out_net1[0].cpu().numpy()
            satellite_feats_jfl = out_net1[1].cpu().numpy()
            synthetic_satellite_feats_jfl = out_net1[2].cpu().numpy()

            ground_features_jfl.append(ground_feats_jfl)
            satellite_features_jfl.append(satellite_feats_jfl)
            synthetic_satellite_features_jfl.append(synthetic_satellite_feats_jfl)

            # Estrazione delle features per il metodo FE
            ground_concatenated_feats_fe = out_net2[0].cpu().numpy()
            satellite_feats_fe = out_net2[1].cpu().numpy()

            ground_features_fe.append(ground_concatenated_feats_fe)
            satellite_features_fe.append(satellite_feats_fe)

            ground_truth_indices.extend(range(i * len(ground), i * len(ground) + len(ground)))

    # Concatenazione delle features per il metodo JFL
    ground_features_jfl = np.vstack(ground_features_jfl)
    satellite_features_jfl = np.vstack(satellite_features_jfl)
    synthetic_satellite_features_jfl = np.vstack(synthetic_satellite_features_jfl)

    # Concatenazione delle features per il metodo FE
    ground_features_fe = np.vstack(ground_features_fe)
    satellite_features_fe = np.vstack(satellite_features_fe)

    # Calcolo delle distanze per il metodo JFL
    distances_ground_satellite_jfl = np.linalg.norm(ground_features_jfl[:, np.newaxis] - satellite_features_jfl, axis=2)
    distances_ground_synthetic_satellite_jfl = np.linalg.norm(synthetic_satellite_features_jfl[:, np.newaxis] - satellite_features_jfl, axis=2)

    # Calcolo delle distanze per il metodo FE
    distances_ground_concatenated_fe = np.linalg.norm(ground_features_fe[:, np.newaxis] - satellite_features_fe, axis=2)

    print("Distances calculated (JFL):", distances_ground_satellite_jfl.shape, distances_ground_synthetic_satellite_jfl.shape)
    print("Distances calculated (FE):", distances_ground_concatenated_fe.shape)

    # Calcolo del recall per le top-k immagini satellitari per entrambi i metodi
    top_1_recall_ground_satellite_jfl = calculate_recall(distances_ground_satellite_jfl, ground_truth_indices, 1)
    top_10_recall_ground_satellite_jfl = calculate_recall(distances_ground_satellite_jfl, ground_truth_indices, 10)
    top_1_percent_recall_ground_satellite_jfl = calculate_recall(distances_ground_satellite_jfl, ground_truth_indices, int(0.01 * distances_ground_satellite_jfl.shape[1]))

    print(f'Top-1 Recall (JFL) - Ground-Satellite: {top_1_recall_ground_satellite_jfl:.4f}')
    print(f'Top-10 Recall (JFL) - Ground-Satellite: {top_10_recall_ground_satellite_jfl:.4f}')
    print(f'Top-1% Recall (JFL) - Ground-Satellite: {top_1_percent_recall_ground_satellite_jfl:.4f}')

    top_1_recall_synt_satellite_jfl = calculate_recall(distances_ground_synthetic_satellite_jfl, ground_truth_indices, 1)
    top_10_recall_synt_satellite_jfl = calculate_recall(distances_ground_synthetic_satellite_jfl, ground_truth_indices, 10)
    top_1_percent_recall_synt_satellite_jfl = calculate_recall(distances_ground_synthetic_satellite_jfl, ground_truth_indices, int(0.01 * distances_ground_synthetic_satellite_jfl.shape[1]))

    print(f'Top-1 Recall (JFL) - Ground-Satellite: {top_1_recall_synt_satellite_jfl:.4f}')
    print(f'Top-10 Recall (JFL) - Ground-Satellite: {top_10_recall_synt_satellite_jfl:.4f}')
    print(f'Top-1% Recall (JFL) - Ground-Satellite: {top_1_percent_recall_synt_satellite_jfl:.4f}')

    top_1_recall_ground_satellite_fe = calculate_recall(distances_ground_concatenated_fe, ground_truth_indices, 1)
    top_10_recall_ground_satellite_fe = calculate_recall(distances_ground_concatenated_fe, ground_truth_indices, 10)
    top_1_percent_recall_ground_satellite_fe = calculate_recall(distances_ground_concatenated_fe, ground_truth_indices, int(0.01 * distances_ground_concatenated_fe.shape[1]))

    print(f'Top-1 Recall (FE) - Ground-Satellite: {top_1_recall_ground_satellite_fe:.4f}')
    print(f'Top-10 Recall (FE) - Ground-Satellite: {top_10_recall_ground_satellite_fe:.4f}')
    print(f'Top-1% Recall (FE) - Ground-Satellite: {top_1_percent_recall_ground_satellite_fe:.4f}')


def calculate_recall(distances, ground_truth_indices, top_k):
    num_queries = distances.shape[0]
    correct_matches = 0
    
    for i in range(num_queries):
        sorted_indices = np.argsort(distances[i])
        if ground_truth_indices[i] in sorted_indices[:top_k]:
            correct_matches += 1
    
    recall = correct_matches / num_queries
    return recall

def prova(dataloader):
    ground_truth_indices = []
    for i, (ground, sat, synt, _, _, _) in enumerate(dataloader):
        ground_truth_indices.extend(range(i * len(ground), i * len(ground) + len(ground)))

    print("Ground truth indices:", ground_truth_indices)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset"
    valCSV = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/val-19zl.csv"

    validation_dataloader = create_dataset(device, dataset_path, valCSV)
    evaluate(device, validation_dataloader)
    #prova(validation_dataloader)

if __name__ == '__main__':
    main()
