import torch.nn as nn
import torch
import torchvision
from torchvision.models import VGG16_Weights


class VGG16(nn.Module):
    """
    Implementation of VGG16 architecture as proposed by Simonyan et al. in
    "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION".
    """

    def __init__(self, num_classes: int = None,
                 include_classifier_part: bool = True,
                 device: torch.device = None):
        """
        Create a VGG16 network for classification, with num_classes possible labels.
        The model input size is 3x224x224.

        :param num_classes: The number of possible classes (the dimension of the output layer)
        :param include_classifier_part: Whether the model should contain the final MLP for classification.
        """
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, device=device),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, device=device),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, device=device),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, device=device),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, device=device),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, device=device),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if include_classifier_part:
            self.layer6 = nn.Sequential(
                nn.Flatten(start_dim=0),
                nn.Linear(512 * 7 * 7, 4096, device=device),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096, device=device),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes, device=device),
                nn.Softmax(dim=0),
            )
        else:
            self.layer6 = nn.Identity()

    def forward(self, x):
        # out1 = self.layer1(x)
        # out2 = self.layer2(out1)
        # out3 = self.layer3(out2)
        # out4 = self.layer4(out3)
        # out5 = self.layer5(out4)
        # return self.layer6(out5)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return self.layer6(out)

    def load_imagenet_weights_feature_extr(self):
        """
        Loads in the feature extractor part of the current model the weights of the VGG16 model pretrained on ImageNet.
        """
        pretrained_model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        pretrained_feat_extr = list(filter(lambda layer: isinstance(layer, nn.Conv2d), pretrained_model.features))

        this_feat_extr = self.layer1 + self.layer2 + self.layer3 + self.layer4 + self.layer5
        this_feat_extr = list(filter(lambda layer: isinstance(layer, nn.Conv2d), this_feat_extr))

        for ind, layer in enumerate(pretrained_feat_extr):
            this_feat_extr[ind].weight = layer.weight
