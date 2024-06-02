import torch
import torchvision
from torch import nn

from gen_model.feat_extr.vgg import VGG
from san_model.model.san import SAN


class FeatureExtractorLoss:
    """
    A custom loss defined in order to train the JointFeatureLearningNetwork and the FeatureFusionNetwork jointly.
    """

    def __init__(self, loss_net_1: torch.Tensor, loss_net_2: torch.Tensor):
        self.loss_net_1 = loss_net_1
        self.loss_net_2 = loss_net_2

    def backward(self):
        self.loss_net_1.backward()
        self.loss_net_2.backward()

    def item(self):
        return self.loss_net_1.item() + self.loss_net_2.item()


class FeatureExtractor(nn.Module):
    def __init__(self, device):
        """
        Create the network to extract features from a triple of ground, satellite and synthetic satellite images,
        as described in "Bridging the Domain Gap for Ground-to-Aerial Image Matching" by Regmi et al.
        This class defines the Joint Feature Learning and Feature Fusion parts
        """
        super(FeatureExtractor, self).__init__()

        self.joint_feature_learning_net = JointFeatureLearningNetwork(device=device)
        self.feature_fusion_net = FeatureFusionNetwork(device=device)

    @staticmethod
    def triplet_loss(output_tuple):
        out_net1, out_net2 = output_tuple

        loss_net_1 = JointFeatureLearningNetwork.triplet_loss(out_net1)
        loss_net_2 = FeatureFusionNetwork.triplet_loss(out_net2)

        return FeatureExtractorLoss(loss_net_1, loss_net_2)

    # make class callable with braces
    def __call__(self, ground, satellite, synthetic_satellite):
        return self.forward(ground, satellite, synthetic_satellite)

    def load_weights(self, path_to_jfl_weights, path_to_ff_weights):
        self.joint_feature_learning_net.load_state_dict(torch.load(path_to_jfl_weights))
        self.feature_fusion_net.load_state_dict(torch.load(path_to_ff_weights))

    def forward(self, ground, satellite, synthetic_satellite):
        """
        Makes a forward pass in the whole network, and returns a tuple that at position 0
        contains the output of the Joint Feature Learning network and at position 1 the output of the Feature
        Fusion network.
        """
        out_net1 = self.joint_feature_learning_net(ground, satellite, synthetic_satellite)
        # Remove gradient information from the tensors passed to network 2 in order to make the learning
        # process of the 2 networks independent of each other.
        inp_net2 = tuple(tensor.detach().clone() for tensor in out_net1)
        out_net2 = self.feature_fusion_net(*inp_net2)

        return out_net1, out_net2


class JointFeatureLearningNetwork(nn.Module):
    def __init__(self, device):
        super(JointFeatureLearningNetwork, self).__init__()

        self.ground_vgg = VGG(device=device, ground_padding=True)
        self.sat_vgg = VGG(device=device)
        # self.sat_gan_vgg = VGG(device=device)
        self.sat_gan_vgg = self.sat_vgg  # enforce weight sharing

        # reference https://github.com/kregmi/cross-view-image-matching/blob/master/joint_feature_learning/src/siamese_fc.py#L57
        self.ground_linear = nn.Linear(in_features=53760, out_features=1000, device=device)
        self.sat_linear = nn.Linear(in_features=43008, out_features=1000, device=device)
        # self.sat_gan_linear = nn.Linear(in_features=43008, out_features=1000, device=device)
        self.sat_gan_linear = self.sat_linear  # enforce weight sharing

    @staticmethod
    def triplet_loss(output_tuple):
        gr_lin_out, sat_lin_out, sat_gan_lin_out = output_tuple

        distance1 = 2 - 2 * (sat_lin_out @ gr_lin_out.T)
        loss1 = SAN.triplet_loss(distance1)
        distance2 = 2 - 2 * (sat_gan_lin_out @ sat_lin_out.T)
        loss2 = SAN.triplet_loss(distance2)

        return (10 * loss1 + loss2) / 11

    def forward(self, ground, satellite, synthetic_satellite):
        ground = torchvision.transforms.Resize((224, 1232))(ground)
        satellite = torchvision.transforms.Resize((512, 512))(satellite)
        synthetic_satellite = torchvision.transforms.Resize((512, 512))(synthetic_satellite)

        ground_vgg_out = self.ground_vgg(ground)
        sat_vgg_out = self.sat_vgg(satellite)
        sat_gan_vgg_out = self.sat_gan_vgg(synthetic_satellite)

        gr_lin_out = self.ground_linear(ground_vgg_out)
        gr_lin_out = nn.functional.normalize(gr_lin_out, p=2, dim=1)

        sat_lin_out = self.sat_linear(sat_vgg_out)
        sat_lin_out = nn.functional.normalize(sat_lin_out, p=2, dim=1)

        sat_gan_lin_out = self.sat_gan_linear(sat_gan_vgg_out)
        sat_gan_lin_out = nn.functional.normalize(sat_gan_lin_out, p=2, dim=1)

        return gr_lin_out, sat_lin_out, sat_gan_lin_out


class FeatureFusionNetwork(nn.Module):
    def __init__(self, device):
        super(FeatureFusionNetwork, self).__init__()

        # reference https://github.com/kregmi/cross-view-image-matching/blob/master/feature_fusion/src/siamese_fc.py#L25
        self.linear_sat = nn.Linear(1000, 1000, device=device)
        self.linear_concat = nn.Linear(2000, 1000, device=device)

    @staticmethod
    def triplet_loss(output_tuple):
        sat_lin_out, conc_lin_out = output_tuple

        distance1 = 2 - 2 * (sat_lin_out @ conc_lin_out.T)
        return SAN.triplet_loss(distance1)

    def forward(self, ground_feats, satellite_feats, synthetic_satellite_feats):
        """
        Run a forward pass in the Feature Fusion network.
        :param ground_feats: The output of the Joint Feature Learning Network for the ground image input.
        :param satellite_feats: The output of the Joint Feature Learning Network for the satellite image input.
        :param synthetic_satellite_feats: The output of the Joint Feature Learning Network for the synthetic
         satellite image input.
        """
        ground_conc_gan_sat = torch.concatenate([ground_feats, synthetic_satellite_feats], dim=1)
        conc_lin_out = self.linear_concat(ground_conc_gan_sat)
        conc_lin_out = nn.functional.normalize(conc_lin_out, p=2, dim=1)

        sat_lin_out = self.linear_sat(satellite_feats)
        sat_lin_out = nn.functional.normalize(sat_lin_out, p=2, dim=1)

        return sat_lin_out, conc_lin_out
