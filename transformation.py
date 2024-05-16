import numpy as np
import torch
import torch.nn.functional as F

class Transformation:
    def __init__(self, name, aerial_size=0, height=0, width=0):
        """

        :param name: name of the transformation for more clarity in the main.py
        :param aerial_size: Size of the aerial image
        :param height: Height of the polar transformed aerial image
        :param width: Width of the polar transformed aerial image
        """
        self.name = name
        self.aerial_size = aerial_size
        self.height = height
        self.width = width

    def sample(self, img, x, y, bounds):
        """
        Sample the image at given coordinates, handling out-of-bounds values.
        
        Args:
        - img: Numpy array with dimensions (H, W, C)
        - x, y: Coordinates to sample at (Numpy arrays)
        - bounds: Bounds of the image (tuple of 4 values)
        
        Returns:
        - Sampled values as a Numpy array
        """
        x0, x1, y0, y1 = bounds
        idxs = (x0 <= x) & (x < x1) & (y0 <= y) & (y < y1)
    
        sample = np.zeros((x.shape[0], x.shape[1], img.shape[-1]))
        sample[idxs, :] = img[x[idxs], y[idxs], :]

        return sample

    def polar(self, img):
        """
        Compute the polar transformation of an aerial image.
        
        Args:
        - img: aerial image with dimensions (H, W, C)
        
        Returns:
        - Polar image with transformed dimensions
        """
        img_width = img.shape[1]
        img_height = img.shape[0]

        i = np.arange(0, self.height)
        j = np.arange(0, self.width)

        jj, ii = np.meshgrid(j, i)  # meshgrid

        y = self.aerial_size / 2. - self.aerial_size / 2. / self.height * (self.height - 1 - ii) * np.sin(
            2 * np.pi * jj / self.width)
        x = self.aerial_size / 2. + self.aerial_size / 2. / self.height * (self.height - 1 - ii) * np.cos(
            2 * np.pi * jj / self.width)

        # Apply bilinear interpolation
        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        # Bounds
        bounds = (0, img_width, 0, img_height)

        img_00 = self.sample(img, x0, y0, bounds)
        img_01 = self.sample(img, x0, y1, bounds)
        img_10 = self.sample(img, x1, y0, bounds)
        img_11 = self.sample(img, x1, y1, bounds)

        na = np.newaxis

        # linear interpolation in x direction
        img_x0 = (x1 - x)[..., na] * img_00 + (x - x0)[..., na] * img_10
        img_x1 = (x1 - x)[..., na] * img_01 + (x - x0)[..., na] * img_11

        # linear interpolation in y direction
        img_xy = (y1 - y)[..., na] * img_x0 + (y - y0)[..., na] * img_x1

        # Convert the polar image to a tensor
        img_xy_tensor = torch.tensor(img_xy, dtype=torch.float32)

        return img_xy_tensor

    def correlation(self, Fs, Fg):
        """
        Compute the correlation between two feature maps Fs and Fg.
        
        Args:
        - Fs: batch of aerial feature maps with dimensions (batch_size, C, H, Ws)
        - Fg: batch of ground feature maps with dimensions (batch_size, C, H, Wv)
        
        Returns:
        - Correlation scores with dimensions (batch_size, Ws, batch_size)
        - Orientation indices with maximum correlation with dimensions (batch_size, batch_size)
        """
        batch_size_s, Cs, Hs, Ws = Fs.shape
        batch_size_g, Cg, Hg, Wv = Fg.shape

        assert Hs == Hg and Cs == Cg, "Fs and Fg must have the same height and number of channels"

        # Warp and pad columns of Fs
        n = Wv - 1
        Fs_padded = torch.cat([Fs, Fs[:, :, :, :n]], dim=3)  # Shape: (batch_size_s, C, H, Ws + n)
        
        # Perform convolution (correlation)
        Fg_transposed = Fg.permute(1, 2, 3, 0)  # Shape: (C, H, Wv, batch_size_g)
        out = F.conv2d(Fs_padded, Fg_transposed, stride=1, padding=0)  # Shape: (batch_size_s, batch_size_g, 1, Ws)

        assert out.shape[2] == 1 and out.shape[3] == Ws, "Output shape is incorrect"
        
        # Remove the height dimension (squeeze)
        out = out.squeeze(2)  # Shape: (batch_size_s, batch_size_g, Ws)
        out = out.permute(0, 2, 1)  # Shape: (batch_size_s, Ws, batch_size_g)
        
        # Convert to numpy array
        correlation_scores = out.detach().cpu().numpy()  # Shape: (batch_size_s, Ws, batch_size_g)

        # Calculate the orientation with the maximum correlation
        orientation_indices = np.argmax(correlation_scores, axis=1)  # Shape: (batch_size_s, batch_size_g)

        orientation_indices_tensor = torch.tensor(orientation_indices)

        # Cast the indices to int32
        orientation_indices_int32 = orientation_indices_tensor.to(dtype=torch.int32)
        
        return correlation_scores, orientation_indices_int32
    
    def torch_shape(x, rank):
        static_shape = list(x.size())
        dynamic_shape = list(x.shape)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape[:rank])]

    def shift_crop(self, Fs, orientation, Wg):
        """
        Shifts and crops the aerial feature maps based on the similarity matrix to align them with the ground image.
        
        Args:
        - Fs: Aerial feature maps
        - similarity_matrix: Similarity matrix between the compared images
        - shift_amount: Amount of shift in pixels
        
        Returns:
        - Shifted and cropped aerial feature maps
        """


        batchS, batchG = self.torch_shape(orientation, 2) # batch dimensions

        batch_size_s, Cs, Hs, Ws = Fs.shape

        Fs = torch.unsqueeze(Fs, 1)  # Add a dimension, shape: (batchS, 1, Cs, Hs, Ws)

        # Tile and permute
        Fs = Fs.repeat(1, batchG, 1, 1, 1)
        Fs = Fs.permute(0, 1, 2, 4, 3)  # shape = [batchS, batchG, Cs, Ws, Hs]

        orientation = torch.unsqueeze(orientation, -1) # Add a dimension, shape: (batchS, batchG, 1)

        i = torch.arange(batchS)
        j = torch.arange(batchG)
        k = torch.arange(Ws)

        # Create meshgrid
        x, y, z = torch.meshgrid(i, j, k)
        z_index = torch.fmod(z + orientation, Ws)  # Compute the module element-wise

        x1 = x.view(-1)
        y1 = y.view(-1)
        z1 = z_index.view(-1)
        index = torch.stack([x1, y1, z1], dim=1)

        extracted_values = torch.gather(Fs, 1, index.unsqueeze(-1).expand(-1, -1, -1, -1, Fs.size(-1)))
        sat = extracted_values.view(batchS, batchG, Cs, Ws, Hs)
        index1 = torch.arange(Wg)

        sat_transposed = sat.permute(4, 1, 0, 3, 2)  # shape: [Ws, batch_sat, batch_grd, h, channel]
        extracted_values = torch.gather(sat_transposed, 0, index1.unsqueeze(-1).expand(-1, -1, -1, -1, -1))
        sat_crop_matrix = extracted_values.permute(1, 2, 4, 3, 0)  # shape: [batch_sat, batch_grd, channel, h, Wg]

        assert sat_crop_matrix.shape[3] == Wg, "The width of the cropped aerial image is incorrect"

        return sat_crop_matrix
    
    def corr_crop_distance(self, Fs, Fg):
        corr_out, corr_orien = self.correlation(Fs, Fg)
        sat_cropped = self.shift_crop(Fs, corr_orien, Fg.shape[3])
        # shape = [batch_sat, batch_grd, channel, h, grd_width]

        sat_matrix = F.normalize(sat_cropped, p=2, dim=(2, 3, 4))
        distance = 2 - 2 * torch.sum(sat_matrix * Fg.unsqueeze(0), dim=(2, 3, 4)).T # shape = [batch_grd, batch_sat]

        return sat_matrix, distance, corr_orien
