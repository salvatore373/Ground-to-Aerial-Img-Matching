from typing import Type

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    """Helper class for model's training and evaluation."""

    def __init__(self, model: nn.Module, ):
        self.model = model

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

                pred = self.model(ground, polar_sat, polar_segm_sat)

                loss = loss_function(pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'
                      .format(epoch, step, train_loss / (step + 1)))

            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, train_loss / len(train_dataloader)))

            # Validation
            self.model.eval()
            with torch.no_grad():
                for polar_sat, polar_segm_sat, ground, _, _, _ in valid_dataloader:
                    pred = self.model(ground, polar_sat, polar_segm_sat)
                    loss = loss_function(pred)
                    valid_loss += loss.item()
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
