import torch
from torch import nn
import tensorflow as tf


class VGG(nn.Module):
    def __init__(self, ground_padding: bool = False, device=None):
        """
        Create a VGG network with the structure described in
        "Bridging the Domain Gap for Ground-to-Aerial Image Matching" by Regmi et al.

        The input should be a batch: shape (batch_size, channels, height, width)
        :param ground_padding: Whether to use the padding necessary when the input is an image of shape
         (channels, 224, 1232).
        """
        super(VGG, self).__init__()
        # reference: https://github.com/kregmi/cross-view-image-matching/blob/master/joint_feature_learning/src/VGG.py#L83

        kernel_size = 4
        stride = (2, 2)
        # padding = 65  # required value to have input equal to output size
        padding = 1
        dropout = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                      padding=(1, 2) if ground_padding else padding, stride=stride,
                      # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=, stride=stride,
                      device=device),
            nn.ReLU(),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                      padding=(2, 2) if ground_padding else padding, stride=stride,
                      device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=padding, stride=stride,
                      device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.init_conv_weights()

    def init_conv_weights(self):
        """
        Initializes the weights of the convolutional layers using Xavier initialization.
        """
        layers = self.layer1 + self.layer2 + self.layer3 + self.layer4
        layers = list(filter(lambda layer: isinstance(layer, nn.Conv2d), layers))

        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # out1 = self.layer1(x)
        # out2 = self.layer2(out1)
        # out3 = self.layer3(out2)
        # out4 = self.layer4(out3)
        # out5 = self.layer5(out4)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out6 = self.layer6(out)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)

        batch_size = x.shape[0]
        return torch.concat(
            [out6.reshape(batch_size, -1), out7.reshape(batch_size, -1), out8.reshape(batch_size, -1)], dim=1)
