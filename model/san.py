import torch
import torchvision.transforms
from torch import nn

from model.vgg import VGG16
from transformation import Transformation


class SAN(nn.Module):
    def __init__(self, input_is_transformed):
        """
        Create a SAN as defined in the paper by Pro et al. "https://arxiv.org/pdf/2404.11302"

        :param input_is_transformed: Whether the satellite_view, satellite_segmented passed to forward()
        have already been polarized.
        """
        super(SAN, self).__init__()

        self.input_is_transformed = input_is_transformed
        self.vgg_ground = VGG16(include_classifier_part=False, circular=False)
        self.vgg_sat = VGG16(include_classifier_part=False, circular=True)
        self.vgg_sat_segm = VGG16(include_classifier_part=False, circular=True)

        # The original size of the aerial image (square)
        self.aerial_imgs_size = 370
        # Height of polar transformed img
        self.polar_height = 128
        # Width of polar transformed img
        self.polar_width = 512

        self.img_processor = Transformation('_', self.aerial_imgs_size, self.polar_height, self.polar_width)

    def triplet_loss(self, distance_matrix):
        #reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/train_no_session.py#L66
        pos_dist = torch.diagonal(distance_matrix, dim1=0, dim2=1)
        # pos_dist = tf.linalg.tensor_diag_part(dist_array)

        ground_to_sat_dist = pos_dist - distance_matrix

    def forward(self, ground_view, satellite_view, satellite_segmented):
        # reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/train_no_session.py#L84
        if not self.input_is_transformed:
            satellite_view = self.img_processor.polar(satellite_view)
            satellite_segmented = self.img_processor.polar(satellite_segmented)

        # DEBUG
        ground_view = torchvision.transforms.Resize((128, 512))(ground_view)

        # Pass the ground view to VGG
        vgg_ground_out = self.vgg_ground(ground_view)

        # Pass the satellite view and its segmentation to the VGGs
        vgg_sat_out = self.vgg_sat(satellite_view)
        vgg_sat_segm_out = self.vgg_sat_segm(satellite_segmented)
        # Concatenate their results
        concat = torch.cat((vgg_sat_out, vgg_sat_segm_out), dim=0 if len(vgg_sat_out.size()) == 3 else 1)

        correlation_res = self.img_processor.correlation(vgg_ground_out, concat)        # reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/cir_net_FOV_mb.py#L48
        print(correlation_res)