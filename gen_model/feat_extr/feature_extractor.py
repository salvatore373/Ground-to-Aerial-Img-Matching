from torch import nn


class FeatureExtractor:
    def __init__(self):
        """
        Create the network to extract features from a triple of ground, satellite and synthetic satellite images,
        as described in "Bridging the Domain Gap for Ground-to-Aerial Image Matching" by Regmi et al.
        This class defines the Joint Feature Learning and Feature Fusion parts
        :param args:
        :param kwargs:
        """
        super(FeatureExtractor, self).__init__()

        # todo: init 2 instances of Joint... and FeatureFusion...

    def forward(self, x):
        pass


class JointFeatureLearningNetwork(nn.Module):
    def __init__(self):
        super(JointFeatureLearningNetwork, self).__init__()

    def forward(self, x):
        pass


class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super(FeatureFusionNetwork, self).__init__()

    def forward(self, x):
        pass