from typing import Type

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    """Helper class for model's training and evaluation."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def train(self,
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader,
              loss_function: callable,
              optimizer: Type[optim.Optimizer] = optim.Adam,
              epochs: int = 1,
              early_stopping_patience: int = 10,
              learning_rate: float = 0.001,
              weight_decay: float = 0.01):

        # Early-stopping params
        patience = early_stopping_patience
        epochs_wtout_impr = 0
        min_val_loss = float('inf')

        # progress_bar = tqdm(range(1, epochs + 1))
        optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(1, epochs + 1):
            print('\033[92m' + ' Epoch {:2d}'.format(epoch) + '\033[0m')

            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            # for batch in train_dataloader:
            for step, (polar_sat, polar_segm_sat, ground, _, _, _) in enumerate(train_dataloader):
                optimizer.zero_grad()

                # Load imgs on GPU
                polar_sat, polar_segm_sat, ground = polar_sat.to(self.device), polar_segm_sat.to(
                    self.device), ground.to(self.device)

                pred = self.model(ground, polar_sat, polar_segm_sat)

                loss = loss_function(pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Remove images from GPU
                polar_sat.detach()
                polar_segm_sat.detach()
                ground.detach()
                del polar_sat
                del polar_segm_sat
                del ground
                torch.cuda.empty_cache()

                print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'
                      .format(epoch, step, train_loss / (step + 1)))

            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, train_loss / len(train_dataloader)))

            # Validation
            self.model.eval()
            with torch.no_grad():
                for polar_sat, polar_segm_sat, ground, _, _, _ in valid_dataloader:
                    # Load imgs on GPU
                    polar_sat, polar_segm_sat, ground = polar_sat.to(self.device), polar_segm_sat.to(
                        self.device), ground.to(self.device)
                    pred = self.model(ground, polar_sat, polar_segm_sat)
                    loss = loss_function(pred)
                    valid_loss += loss.item()
                    # Remove images from GPU
                    polar_sat.detach()
                    polar_segm_sat.detach()
                    ground.detach()
                    del polar_sat
                    del polar_segm_sat
                    del ground
                    torch.cuda.empty_cache()
            valid_loss /= len(valid_dataloader)
            print('\t[E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

            # Early-stopping
            if valid_loss >= min_val_loss:
                epochs_wtout_impr += 1
            else:
                epochs_wtout_impr = 0
            if epochs_wtout_impr >= patience:
                print('Training stopped by early stopping.')
                break
            min_val_loss = min(min_val_loss, valid_loss)

    def evaluate(self, test_dataloader: DataLoader, batch_size, features_output_dim: tuple,
                 model_output_filter: callable = None):
        num_samples = batch_size * len(test_dataloader)
        grd_accumulator = np.zeros([num_samples, *features_output_dim])
        sat_accumulator = np.zeros([num_samples, *features_output_dim])
        accumulator_ind = 0

        self.model.eval()
        with torch.no_grad():
            for polar_sat, polar_segm_sat, ground, _, _, _ in test_dataloader:
                # Load imgs on GPU
                polar_sat, polar_segm_sat, ground = polar_sat.to(self.device), polar_segm_sat.to(
                    self.device), ground.to(self.device)

                if model_output_filter is None:
                    dist_mat_pred, vgg_ground_out, sat_vgg_concat = self.model(ground, polar_sat, polar_segm_sat,
                                                                               return_features=True)
                if model_output_filter is not None:
                    model_output = self.model(ground, polar_sat, polar_segm_sat)
                    dist_mat_pred, vgg_ground_out, sat_vgg_concat = model_output_filter(model_output)

                grd_accumulator[accumulator_ind:accumulator_ind + batch_size, :] = vgg_ground_out
                sat_accumulator[accumulator_ind:accumulator_ind + batch_size, :] = sat_vgg_concat

                accumulator_ind += batch_size

                # Remove images from GPU
                polar_sat.detach()
                polar_segm_sat.detach()
                ground.detach()
                del polar_sat
                del polar_segm_sat
                del ground
                torch.cuda.empty_cache()

        # Flatten features and compute distance
        grd_accumulator = grd_accumulator.reshape(num_samples, -1)
        sat_accumulator = sat_accumulator.reshape(num_samples, -1)
        sat_accumulator /= np.linalg.norm(sat_accumulator, axis=-1, keepdims=True)
        dist_array = 2 - 2 * np.matmul(grd_accumulator, np.transpose(sat_accumulator))

        # Compute the accuracy
        pos_samples = 0
        topK = 1
        for i in range(num_samples):
            ground_truth_dist = dist_array[i, i]
            false_pos = np.sum(dist_array[i, :] < ground_truth_dist)
            if false_pos < topK:
                # The ground image has been correctly matched to the ground truth with the lowest distance
                pos_samples += 1
        return pos_samples / num_samples  # accuracy
