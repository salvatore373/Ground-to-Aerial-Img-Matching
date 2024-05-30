import torch
from torch import optim
from torch.utils.data import RandomSampler, DataLoader

from gen_model.feat_extr.feature_extractor import FeatureExtractor
from san_model.model.data import CrossViewDataset, ImageTypes
from san_model.model.trainer import Trainer


def train_joint_feature_learner(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train-19zl.csv"

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

def test_network():
    sat_img = torch.rand(10, 3, 512, 512)
    grnd_img = torch.rand(10, 3, 224, 1232)  # if change size, change ground_padding
    sat_synth_img = torch.rand(10, 3, 512, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vgg_model = VGG(device)
    # res = vgg_model(rand_img)

    res = FeatureExtractor(device)(grnd_img, sat_img, sat_synth_img)
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_network()
    # eight_layer_conv_multiscale()
    # three_stream_joint_feat_learning()

    train_joint_feature_learner(device)


if __name__ == '__main__':
    main()
