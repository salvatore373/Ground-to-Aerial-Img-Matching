import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import VGG16_Weights


class _ConcatPadLayer(nn.Module):
    def __init__(self, n):
        """
        Build a layer that performs tensor concatenation and padding for circular convolutions
        """
        super(_ConcatPadLayer, self).__init__()
        self.n = n

    def forward(self, x):
        if len(x.size()) == 4:
            out = torch.cat((x[:, :, -self.n:, :], x, x[:, :, :self.n, :]), dim=2)
            out = F.pad(out, (self.n, self.n, 0, 0))
            return out
        if len(x.size()) == 3:
            out = torch.cat((x[:, -self.n:, :], x, x[:, :self.n, :]), dim=1)
            out = F.pad(out, (self.n, self.n, 0, 0))
            return out


class VGG16(nn.Module):
    """
    Implementation of VGG16 architecture as proposed by Simonyan et al. in
    "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION", in the modified version proposed by
    "A SEMANTIC SEGMENTATION-GUIDED APPROACH FOR GROUND-TO-AERIAL IMAGE MATCHING" by Pro et al.
    """

    def __init__(self, num_classes: int = None,
                 include_classifier_part: bool = True,
                 circular: bool = False,
                 device: torch.device = None):
        """
        Create a VGG16 network for classification, with num_classes possible labels.
        The original VGG16 input size is 3x224x224.

        :param num_classes: The number of possible classes (the dimension of the output layer)
        :param circular: Whether the VGG16 should perform circular convolutions.
        :param include_classifier_part: Whether the model should contain the final MLP for classification.
        """
        super(VGG16, self).__init__()
        self.circular = circular

        # shape comments to be considered for input shape (1, 3, 128, 512)

        self.concat_and_pad = _ConcatPadLayer(n=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', device=device),
            nn.ReLU(),
            # only in VGG std arhitecture nn.MaxPool2d(kernel_size=2, stride=2)
        )  # shape: 1, 512, 16, 64

        # layer 5 as in standard VGG architecture
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', device=device),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', device=device),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', device=device),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        # layer 5 as in modified VGG architecture
        if self.circular:
            self.layer5 = nn.Sequential(
                self.concat_and_pad,  # shape: 1, 512, 18, 66
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding='valid', stride=(2, 1),
                          device=device),  # shape: 1, 256, 8, 64
                nn.ReLU(),
                self.concat_and_pad,  # shape: 1, 256, 10, 66
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='valid', stride=(2, 1),
                          device=device),  # shape: 1, 64, 4, 64
                nn.ReLU(),
                self.concat_and_pad,  # shape: 1, 64, 6, 66
                nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding='valid', stride=1,
                          device=device),  # shape: 1, 8, 4, 64
                nn.ReLU(),
            )
        else:
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=(1, 1), stride=(2, 1),
                          device=device),  # 256, 8, 64
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=(1, 1), stride=(2, 1),
                          device=device),  # 64, 4, 64
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding='same', stride=1,
                          device=device),  # 16, 4, 64
                nn.ReLU(),
            )

        # Set layers 1, 2, 3 not trainable, as in modified VGG architecture
        self.layer1.requires_grad = False
        self.layer2.requires_grad = False
        self.layer3.requires_grad = False

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

        this_feat_extr = self.layer1 + self.layer2 + self.layer3 + self.layer4
        this_feat_extr = list(filter(lambda layer: isinstance(layer, nn.Conv2d), this_feat_extr))

        for ind in range(len(this_feat_extr)):
            this_feat_extr[ind].weight = pretrained_feat_extr[ind].weight
