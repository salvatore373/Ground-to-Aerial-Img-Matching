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

    def triplet_loss(self, distance_matrix: torch.Tensor, loss_weight: float = 10.0) -> torch.Tensor:
        """
        Compute the triplet loss with soft margin with the given distances between samples.
        :param distance_matrix: A matrix of distances between samples, with shape batch_gr, batch_sat.
        :param loss_weight: The factor to multiply distances.
        :return: The value of the computed loss
        """
        # reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/train_no_session.py#L66
        # Get distances of positive pairs
        pos_dist = torch.diagonal(distance_matrix, dim1=0, dim2=1)
        num_pairs = distance_matrix.size()[0] * (distance_matrix.size()[0] - 1)

        ground_to_sat_dist = pos_dist - distance_matrix  # dist mat without main diag
        loss_gr_to_sat = torch.sum(torch.log(1 + torch.exp(ground_to_sat_dist * loss_weight))).div(num_pairs)

        sat_to_ground_dist = torch.unsqueeze(pos_dist,
                                             1) - distance_matrix  # each col of dist_mat subtracted by pos_dist
        loss_sat_to_gr = torch.sum(torch.log(1 + torch.exp(sat_to_ground_dist * loss_weight))).div(num_pairs)

        return (loss_gr_to_sat + loss_sat_to_gr) / 2.0

    def tf_correl(self, vgg_gr, sat_concat):
        import tensorflow as tf
        def corr(sat_matrix, grd_matrix):
            s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
            g_h, g_w, g_c = grd_matrix.get_shape().as_list()[1:]
            assert s_h == g_h, s_c == g_c  # devono avere la stessa altezza e lo stesso numero di canali

            def warp_pad_columns(x, n):
                out = tf.concat([x, x[:, :, :n, :]], axis=2)
                return out

            n = g_w - 1

            x = warp_pad_columns(sat_matrix, n)

            # Correlation between Fs and Fg (use convolution but transposing the filter = correlation)
            f = tf.transpose(a=grd_matrix, perm=[1, 2, 3, 0])
            out = tf.nn.conv2d(input=x, filters=f, strides=[1, 1, 1, 1], padding='VALID')
            h, w = out.get_shape().as_list()[1:-1]

            assert h == 1, w == s_w

            out = tf.squeeze(out, axis=1)  # shape = [batch_sat, w, batch_grd]
            # Get the area with the maximum correlation
            orien = tf.argmax(input=out, axis=1)  # shape = [batch_sat, batch_grd]

            return out, tf.cast(orien, tf.int32)

        def tf_shape(x, rank):
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(input=x), rank)
            return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]

        def crop_sat(sat_matrix, orien, grd_width):
            batch_sat, batch_grd = tf_shape(orien, 2)  # ritorna le dimensioni dei due batch
            h, w, channel = sat_matrix.get_shape().as_list()[1:]
            sat_matrix = tf.expand_dims(sat_matrix, 1)  # shape=[batch_sat, 1, h, w, channel]
            sat_matrix = tf.tile(sat_matrix, [1, batch_grd, 1, 1, 1])
            sat_matrix = tf.transpose(a=sat_matrix,
                                      perm=[0, 1, 3, 2, 4])  # shape = [batch_sat, batch_grd, w, h, channel]
            orien = tf.expand_dims(orien, -1)  # shape = [batch_sat, batch_grd, 1]

            i = tf.range(batch_sat)
            j = tf.range(batch_grd)
            k = tf.range(w)

            x, y, z = tf.meshgrid(i, j, k, indexing='ij')
            z_index = tf.math.floormod(z + orien, w)
            x1 = tf.reshape(x, [-1])
            y1 = tf.reshape(y, [-1])
            z1 = tf.reshape(z_index, [-1])
            index = tf.stack([x1, y1, z1], axis=1)
            sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_grd, w, h, channel])
            index1 = tf.range(grd_width)
            sat_crop_matrix = tf.transpose(a=tf.gather(tf.transpose(a=sat, perm=[2, 0, 1, 3, 4]), index1),
                                           perm=[1, 2, 3, 0, 4])
            # shape = [batch_sat, batch_grd, h, grd_width, channel]
            assert sat_crop_matrix.get_shape().as_list()[3] == grd_width

            return sat_crop_matrix

        # sat_vgg = tf.random.uniform(shape=[10, 4, 64, 16], minval=0, maxval=1)
        # grd_vgg = tf.random.uniform(shape=[10, 4, 64, 16], minval=0, maxval=1)
        sat_vgg = sat_concat
        grd_vgg = vgg_gr

        sat_vgg_torch = torch.from_numpy(sat_vgg.numpy())
        grd_vgg_torch = torch.from_numpy(vgg_gr.numpy())
        correlation_out, correlation_orient = self.img_processor.correlation(grd_vgg_torch, sat_vgg_torch)
        cropped_sat = self.img_processor.crop_sat(sat_vgg_torch, correlation_orient, grd_vgg_torch.size()[-1])
        sat_mat = nn.functional.normalize(cropped_sat, p=2, dim=(2, 3, 4))
        distance = 2 - 2 * (torch.sum(sat_mat * grd_vgg_torch.unsqueeze(dim=0), dim=(2, 3, 4))).T

        corr_out, corr_orien = corr(sat_vgg, grd_vgg)
        sat_cropped = crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
        # shape = [batch_sat, batch_grd, h, grd_width, channel]
        sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])
        distance = 2 - 2 * tf.transpose(
            a=tf.reduce_sum(input_tensor=sat_matrix * tf.expand_dims(grd_vgg, axis=0), axis=[2, 3, 4]))
        # shape = [batch_grd, batch_sat]

        # return sat_matrix, distance, corr_orien  # shapes: (1,1,4,64,16), (1,1), (1,1)
        return sat_matrix, distance, corr_orien  # shapes: (10,10,4,64,16), (10,10), (10,10)

    def forward(self, ground_view, satellite_view, satellite_segmented):
        # reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/train_no_session.py#L84
        if not self.input_is_transformed:
            satellite_view = torch.stack([self.img_processor.polar(img) for img in satellite_view.unbind(0)])
            satellite_segmented = torch.stack([self.img_processor.polar(img) for img in satellite_segmented.unbind(0)])

        # DEBUG
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

        # DEBUG
        import tensorflow as tf
        tf1 = tf.convert_to_tensor(vgg_ground_out.detach().numpy())
        tf2 = tf.convert_to_tensor(concat.detach().numpy())
        tf1 = tf.transpose(tf1, perm=[0, 2, 3, 1])
        tf2 = tf.transpose(tf2, perm=[0, 2, 3, 1])
        correlation_res_tf = self.tf_correl(tf1, tf2)

        correlation_out, correlation_orient = self.img_processor.correlation(vgg_ground_out,
                                                                             concat)  # reference https://github.com/pro1944191/SemanticAlignNet/blob/7438b28a78acc821109e08cfa024c52e6143d38f/SAN/cir_net_FOV_mb.py#L48
        cropped_sat = self.img_processor.crop_sat(concat, correlation_orient, vgg_ground_out.size()[-1])
        sat_mat = nn.functional.normalize(cropped_sat, p=2, dim=(2, 3, 4))
        distance = 2 - 2 * (torch.sum(sat_mat * vgg_ground_out.unsqueeze(dim=0), dim=(2, 3, 4))).T
        # return sat_matrix, distance, corr_orien  # shapes: (10,10,4,64,16), (10,10), (10,10)

        return distance
