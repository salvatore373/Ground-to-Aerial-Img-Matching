import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler

from gen_model.feat_extr.feature_extractor import FeatureExtractor
from san_model.model.data import CrossViewDataset, ImageTypes
from san_model.model.trainer import Trainer


def train_feature_extractor(device,
                            dataset_path="/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/",
                            trainCSV="/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train500-19zl.csv"):
    batch_size = 8
    epochs = 30

    full_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                    dataset_content=[ImageTypes.Sat, ImageTypes.SyntheticSat,
                                                     ImageTypes.Ground])
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(0.1 * len(train_dataset)))
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler)

    gen_model = FeatureExtractor(device)
    trainer = Trainer(gen_model, device=device)
    trainer.train(training_dataloader,
                  validation_dataloader,
                  epochs=epochs,
                  loss_function=gen_model.triplet_loss,
                  optimizer=optim.Adam,
                  learning_rate=10e-4, weight_decay=0.01)


def evaluate_feature_extractor(device,
                               dataset_path="/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/",
                               valCSV="/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/val500-19zl.csv"):
    batch_size = 8

    validation_dataset = CrossViewDataset(valCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat,
                                                           ImageTypes.Ground])
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    gen_model = FeatureExtractor(device)
    gen_model.load_weights(
        "/Volumes/SALVATORE R/Università/CV/hw_data/saved_models/models_gen/jfl_1717343102687.pt",
        "/Volumes/SALVATORE R/Università/CV/hw_data/saved_models/models_gen/ff_1717343102687.pt",
    )
    trainer = Trainer(gen_model, device=device)
    print('Starting evaluation...')
    top1Recall = trainer.evaluate(validation_dataloader, batch_size, features_output_dim=(1000,),
                                  model_output_filter=lambda model_output: (None, *model_output[1]))
    print(f'top1Recall: {top1Recall:.4f}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # evaluate_feature_extractor(device)
    train_feature_extractor(device)


if __name__ == '__main__':
    main()
