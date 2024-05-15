import torch
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
        self.vgg_ground = VGG16(include_classifier_part=False)
        self.vgg_sat = VGG16(include_classifier_part=False, circular=True)
        self.vgg_sat_segm = VGG16(include_classifier_part=False, circular=True)

        # The original size of the aerial image (square)
        self.aerial_imgs_size = 370
        # Height of polar transformed img
        self.polar_height = 370
        # Width of polar transformed img
        self.polar_width = 370

        # self.

    def forward(self, ground_view, satellite_view, satellite_segmented):
        if not self.input_is_transformed:
            img_processor = Transformation('polar', self.aerial_imgs_size, self.polar_height, self.polar_width)
            satellite_view = img_processor.polar(satellite_view)
            satellite_segmented = img_processor.polar(satellite_segmented)

        # Pass the satellite view and its segmentation to the VGGs
        vgg_sat_out = self.vgg_sat(satellite_view)
        vgg_sat_segm_out = self.vgg_sat_segm(satellite_segmented)
        # Concatenate their results
        q = torch.cat((vgg_sat_out, vgg_sat_segm_out), dim=0 if len(vgg_sat_out.size()) == 3 else 1)


        pass