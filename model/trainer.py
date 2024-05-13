import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader


# class TrainerForKaggle():
#     def train(self):
#         optimizer = AdamW(model.parameters(), lr=2e-4)
#         num_epochs = CFG.epochs
#         num_training_steps = num_epochs * len(train_dataloader)
#         lr_scheduler = get_scheduler(
#             name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
#         )
#         # Early-stopping params
#         patience = 10
#         epochs_wtout_impr = 0
#         min_val_loss = float('inf')
#
#         progress_bar = tqdm(range(num_training_steps))
#         for epoch in range(num_epochs):
#             train_loss = 0.0
#             valid_loss = 0.0
#
#             # Training
#             model.train()
#             for batch in train_dataloader:
#                 outputs = model(**batch)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
#                 progress_bar.update(1)
#                 train_loss += loss.item()
#                 break
#
#             # Validation
#             model.eval()
#             with torch.no_grad():
#                 for batch_valid in valid_dataloader:
#                     target = model(**batch_valid)
#                     loss = target.loss
#                     valid_loss = loss.item() * batch_valid['input_values'].size(0)
#
#             # Early-stopping
#             val_loss = valid_loss / len(valid_dataloader)
#             if val_loss >= min_val_loss:
#                 epochs_wtout_impr += 1
#             else:
#                 epochs_wtout_impr = 0
#             if epochs_wtout_impr >= patience:
#                 break
#             min_val_loss = min(min_val_loss, val_loss)
#
#             print(
#                 f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_dataloader):.4f} \t\t Validation Loss: {valid_loss / len(valid_dataloader):.4f}')
#
#
# class TrainerFromMNLP():
#     """Utility class to train and evaluate a model."""
#
#     def __init__(
#             self,
#             model: nn.Module,
#             optimizer: torch.optim.Optimizer,
#             log_steps: int = 1_000,
#             log_level: int = 2,
#             loss_function=nn.CrossEntropyLoss(),
#     ):
#         self.model = model
#         self.optimizer = optimizer
#         # self.loss_function = nn.CrossEntropyLoss() # this is the default loss used nearly everywhere in NLP
#         self.loss_function = loss_function
#
#         self.log_steps = log_steps
#         self.log_level = log_level
#
#     def train(
#             self,
#             train_dataloader: DataLoader,
#             valid_dataloader: DataLoader,
#             epochs: int = 1,
#             early_stopping_patience: int = 10,
#     ) -> dict[str, list[float]]:
#         """
#         Args:
#             train_dataloader: a DataLoader instance containing the training instances.
#             valid_dataloader: a DataLoader instance used to evaluate learning progress.
#             epochs: the number of times to iterate over train_dataset.
#
#         Returns:
#             avg_train_loss: the average training loss on train_dataset over epochs.
#         """
#         assert epochs >= 1 and isinstance(epochs, int)
#         if self.log_level > 0:
#             print('Training ...')
#         train_loss = 0.0
#
#         losses = {
#             "train_losses": [],
#             "valid_losses": [],
#             "valid_acc": [],
#         }
#
#         # Early-stopping params
#         patience = early_stopping_patience
#         epochs_wtout_impr = 0
#         min_val_loss = float('inf')
#
#         for epoch in range(1, epochs + 1):
#             if self.log_level > 0:
#                 print('\033[92m' + ' Epoch {:2d}'.format(epoch) + '\033[0m')
#
#             epoch_loss = 0.0
#             self.model.train()
#
#             # for each batch
#             for step, (sequence_lengths, inputs, labels) in enumerate(train_dataloader):
#                 self.optimizer.zero_grad()
#
#                 # We get the predicted logits from the model, with no need to perform any flattening
#                 # as both predictions and labels refer to the whole sentence.
#                 predictions = self.model((sequence_lengths, inputs))
#
#                 # The CrossEntropyLoss expects the predictions to be logits, i.e. non-softmaxed scores across
#                 # the number of classes, and the labels to be a simple tensor of labels.
#                 # Specifically, predictions needs to be of shape [B, C], where B is the batch size and C is the number of
#                 # classes, while labels must be of shape [B] where each element l_i should 0 <= l_i < C.
#                 # See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more information.
#                 sample_loss = self.loss_function(predictions, labels)
#                 sample_loss.backward()
#                 self.optimizer.step()
#
#                 epoch_loss += sample_loss.cpu().tolist()
#
#                 if self.log_level > 1 and (step % self.log_steps) == (self.log_steps - 1):
#                     print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step,
#                                                                                      epoch_loss / (step + 1)))
#
#             avg_epoch_loss = epoch_loss / len(train_dataloader)
#
#             if self.log_level > 0:
#                 print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))
#
#             # perform validation
#             valid_loss, valid_acc = self.evaluate(valid_dataloader)
#
#             # Early-stopping
#             if valid_loss >= min_val_loss:
#                 epochs_wtout_impr += 1
#             else:
#                 epochs_wtout_impr = 0
#             if epochs_wtout_impr >= patience:
#                 print('Training stopped by early stopping.')
#                 break
#             min_val_loss = min(min_val_loss, valid_loss)
#
#             losses["train_losses"].append(avg_epoch_loss)
#             losses["valid_losses"].append(valid_loss)
#             losses["valid_acc"].append(valid_acc)
#
#             if self.log_level > 0:
#                 print('  [E: {:2d}] valid loss = {:0.4f}, valid acc = {:0.4f}'.format(epoch, valid_loss, valid_acc))
#
#             # Communicate results to cross-validator
#             # checkpoint_data = {
#             #     "epoch": epoch,
#             #      "net_state_dict": self.model.state_dict(),
#             #      "optimizer_state_dict": self.optimizer.state_dict(),
#             #  }
#             #  checkpoint = Checkpoint.from_dict(checkpoint_data)
#
#             with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#                 path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
#                 torch.save(
#                     (self.model.state_dict(), self.optimizer.state_dict()), path
#                 )
#                 torch.save(
#                     {"epoch": epoch}, os.path.join(temp_checkpoint_dir, "extra_state.pt"))
#                 checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
#                 train.report(
#                     {"loss": valid_loss, "accuracy": valid_acc},
#                     checkpoint=checkpoint,
#                 )
#
#         if self.log_level > 0:
#             print('... Done!')
#
#         return losses
#
#     def _compute_acc(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
#         # logits [B, 2] are the logits outputted by the BiLSTM model's forward()
#         # We take the argmax along the second dimension (dim=1), so we get a tensor of shape [B]
#         # where each element is 0 if the 0-class had higher logit, 1 otherwise.
#         predictions = torch.argmax(logits, dim=1)
#         # We can then directly compare each prediction with the labels, as they are both tensors with shape [B].
#         # The average of the boolean equality checks between the two is the accuracy of these predictions.
#         # For example, if:
#         #   predictions = [1, 0, 0, 1, 1]
#         #   labels = [1, 0, 1, 1, 1]
#         # The comparison is:
#         #   (predictions == labels) => [1, 1, 0, 1, 1]
#         # which averaged gives an accuracy of 4/5, i.e. 0.80.
#         return torch.mean((predictions == labels).float()).tolist()  # type: ignore
#
#     def computeConfusionMatrix(self, dataloader: DataLoader):
#         tot_labels = torch.tensor([]).cuda()
#         tot_predictions = torch.tensor([]).cuda()
#
#         self.model.eval()
#         with torch.no_grad():
#             for batch in dataloader:
#                 sequence_lengths, inputs, labels = batch
#
#                 logits = self.model((sequence_lengths, inputs))
#                 predictions = torch.argmax(logits, dim=1)
#
#                 tot_labels = torch.cat((tot_labels, labels))
#                 tot_predictions = torch.cat((tot_predictions, predictions))
#
#         return confusion_matrix(tot_labels.cpu().numpy(), tot_predictions.cpu().numpy(), normalize='true')
#
#     def evaluate(self, valid_dataloader: DataLoader) -> tuple[float, float]:
#         """
#         Args:
#             valid_dataloader: the DataLoader to use to evaluate the model.
#
#         Returns:
#             avg_valid_loss: the average validation loss over valid_dataloader.
#         """
#         valid_loss = 0.0
#         valid_acc = 0.0
#         # When running in inference mode, it is required to have model.eval() AND .no_grad()
#         # Among other things, these set dropout to 0 and turn off gradient computation.
#         self.model.eval()
#         with torch.no_grad():
#             for batch in valid_dataloader:
#                 sequence_lengths, inputs, labels = batch
#
#                 logits = self.model((sequence_lengths, inputs))
#
#                 # Same considerations as the training step apply here
#                 sample_loss = self.loss_function(logits, labels)
#                 valid_loss += sample_loss.tolist()
#
#                 sample_acc = self._compute_acc(logits, labels)
#                 valid_acc += sample_acc
#
#         return valid_loss / len(valid_dataloader), valid_acc / len(valid_dataloader),
#
#     def predict(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             x: a tensor of indices
#         Returns:
#             A tuple composed of:
#             - the logits of each class, 0 and 1
#             - the prediction for each sample in the batch
#               0 if the sentiment of the sentence is negative, 1 if it is positive.
#         """
#         self.model.eval()
#         with torch.no_grad():
#             sequence_lengths, inputs = batch
#             logits = self.model(sequence_lengths, inputs)  # [B, 2]
#             predictions = torch.argmax(logits, -1)  # [B, 1] computed on the last dimension of the logits tensor
#             return logits, predictions


# model.train()
# for epoch in range(1, 5):
#     with tqdm(train_loader, unit="batch") as tepoch:
#         for data, target in tepoch:
#             tepoch.set_description(f"Epoch {epoch}")
#
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             predictions = output.argmax(dim=1, keepdim=True).squeeze()
#             loss = F.nll_loss(output, target)
#             correct = (predictions == target).sum().item()
#             accuracy = correct / batch_size
#
#             loss.backward()
#             optimizer.step()
#
#             tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

class Trainer:
    """Helper class for model's training and evaluation."""

    def __init__(self, model: nn.Module, ):
        self.model = model

    def train(self,
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_function,
              epochs: int = 1,
              early_stopping_patience: int = 10):

        # Early-stopping params
        patience = early_stopping_patience
        epochs_wtout_impr = 0
        min_val_loss = float('inf')

        # progress_bar = tqdm(range(1, epochs + 1))
        for epoch in range(1, epochs + 1):
            print('\033[92m' + ' Epoch {:2d}'.format(epoch) + '\033[0m')

            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            # for batch in train_dataloader:
            for step, (sequence_lengths, inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()

                # pred = self.model(**batch)
                pred = self.model((sequence_lengths, inputs))

                # loss = pred.loss
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # train_loss += loss.cpu().tolist()

                # progress_bar.update(1)
                print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'
                      .format(epoch, step, train_loss / (step + 1)))

            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, train_loss / len(train_dataloader)))

            # Validation
            self.model.eval()
            with torch.no_grad():
                for sequence_lengths, inputs, labels in valid_dataloader:
                    pred = self.model((sequence_lengths, inputs))
                    loss = loss_function(pred, labels)
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
