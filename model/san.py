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
        self.vgg_sat = VGG16(include_classifier_part=False)
        self.vgg_sat_segm = VGG16(include_classifier_part=False)

        self.

    def forward(self, ground_view, satellite_view, satellite_segmented):
        if not self.input_is_transformed:
            # TODO: remove hardcoded values
            img_processor = Transformation('polar', 370, 128, 512)
            satellite_view = img_processor.polar(satellite_view)
            satellite_segmented = img_processor.polar(satellite_segmented)



        pass