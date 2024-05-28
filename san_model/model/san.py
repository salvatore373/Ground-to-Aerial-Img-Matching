import torch
import torchvision.transforms
from torch import nn

from san_model.model.vgg import VGG16
from san_model.model.transformation import Transformation


class SAN(nn.Module):
    def __init__(self, input_is_transformed, device: torch.device = None):
        """
        Create a SAN as defined in the paper by Pro et al. "https://arxiv.org/pdf/2404.11302"

        :param input_is_transformed: Whether the satellite_view, satellite_segmented passed to forward()
        have already been polarized.
        """
        super(SAN, self).__init__()

        self.input_is_transformed = input_is_transformed
        self.vgg_ground = VGG16(include_classifier_part=False, circular=False, device=device)
        self.vgg_sat = VGG16(include_classifier_part=False, circular=True, device=device)
        self.vgg_sat_segm = VGG16(include_classifier_part=False, circular=True, device=device)

        # The original size of the aerial image (square)
        self.aerial_imgs_size = 370
        # Height of polar transformed img
        self.polar_height = 128
        # Width of polar transformed img
        self.polar_width = 512

        self.img_processor = Transformation('_', self.aerial_imgs_size, self.polar_height, self.polar_width)

    def triplet_loss(self, distance_matrix: torch.Tensor, loss_weight: float = 10.0) -> torch.Tensor:
        """
        Compute the triplet loss with soft margin with the given distances between samples.
        :param distance_matrix: A matrix of distances between samples, with shape batch_gr, batch_sat.
        :param loss_weight: The factor to multiply distances.
        :return: The value of the computed loss
        """
        # Get distances of positive pairs
        pos_dist = torch.diagonal(distance_matrix, dim1=0, dim2=1)
        num_pairs = distance_matrix.size()[0] * (distance_matrix.size()[0] - 1)

        ground_to_sat_dist = pos_dist - distance_matrix  # dist mat without main diag
        loss_gr_to_sat = torch.sum(torch.log(1 + torch.exp(ground_to_sat_dist * loss_weight))).div(num_pairs)

        sat_to_ground_dist = torch.unsqueeze(pos_dist,
                                             1) - distance_matrix  # each col of dist_mat subtracted by pos_dist
        loss_sat_to_gr = torch.sum(torch.log(1 + torch.exp(sat_to_ground_dist * loss_weight))).div(num_pairs)

        return (loss_gr_to_sat + loss_sat_to_gr) / 2.0

    def forward(self, ground_view, satellite_view, satellite_segmented, return_features: bool = False):
        if not self.input_is_transformed:
            satellite_view = torch.stack([self.img_processor.polar(img) for img in satellite_view.unbind(0)])
            satellite_segmented = torch.stack([self.img_processor.polar(img) for img in satellite_segmented.unbind(0)])

        ground_view = torchvision.transforms.Resize((128, 512))(ground_view)
        satellite_view = torchvision.transforms.Resize((128, 512))(satellite_view)
        satellite_segmented = torchvision.transforms.Resize((128, 512))(satellite_segmented)

        # Pass the ground view to VGG
        vgg_ground_out = self.vgg_ground(ground_view)

        # Pass the satellite view and its segmentation to the VGGs
        vgg_sat_out = self.vgg_sat(satellite_view)
        vgg_sat_segm_out = self.vgg_sat_segm(satellite_segmented)
        # Concatenate their results
        concat = torch.cat((vgg_sat_out, vgg_sat_segm_out), dim=0 if len(vgg_sat_out.size()) == 3 else 1)

        # Compute the distance matrix
        correlation_out, correlation_orient = self.img_processor.correlation(vgg_ground_out, concat)
        cropped_sat = self.img_processor.crop_sat(concat, correlation_orient, vgg_ground_out.size()[-1])
        sat_mat = nn.functional.normalize(cropped_sat, p=2, dim=(2, 3, 4))
        distance = 2 - 2 * (torch.sum(sat_mat * vgg_ground_out.unsqueeze(dim=0), dim=(2, 3, 4))).T

        if return_features:
            return distance, vgg_ground_out, concat
        else:
            return distance
