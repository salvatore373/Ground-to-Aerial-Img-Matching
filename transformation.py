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
        - img: aerial image with dimensions (C, H, W)
        
        Returns:
        - Polar image with transformed dimensions
        """
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        else:
            img = np.transpose(img, (1, 2, 0))
            
        img_width = img.shape[0]
        img_height = img.shape[1]

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
        img_xy_tensor = img_xy_tensor.permute(2, 0, 1)

        return img_xy_tensor

    def corr(self, sat_matrix, grd_matrix):
        # matrix shape
        
        s_n, s_c, s_h, s_w = sat_matrix.shape
        g_n, g_c, g_h, g_w = grd_matrix.shape

        print(f"Shape of satellite matrix: {sat_matrix.shape}, Shape of ground matrix: {grd_matrix.shape}")
        
        assert s_h == g_h and s_c == g_c, "Le matrici devono avere la stessa altezza e lo stesso numero di canali"
        
        def warp_pad_columns(x, n):
            return torch.cat([x, x[:, :, :, :n]], dim=3)
        
        n = g_w - 1
        x = warp_pad_columns(sat_matrix, n) # shape: [batch_size_sat, channels, height, width + n]
        print("Padded satellite matrix: ", x.shape)

        # Correlation
        f = grd_matrix.permute(1, 2, 3, 0)  # Shape: (C, H, W, batch_size_grd)
        f = f.contiguous().view(g_c, g_h, g_w, -1) 
        f = f.permute(3, 0, 1, 2)  # Shape: (batch_size_grd, C, H, W)
        print("Filter matrix: ", f.shape)
        
        # Convolution
        out = F.conv2d(x, f, stride=1, padding=0)  # Shape: (batch_size_sat, batch_size_grd, 1, w)
        print(f"Convolution output: {out.shape}")
        
        out = out.view(s_n, g_n, 1, s_w)  # Reshape
        h, w = out.shape[2:]
        print("Reshaped output: ", out.shape)

        assert h == 1 and w == s_w, "L'altezza del risultato deve essere 1 e la larghezza deve corrispondere a quella di sat_matrix"
        
        out = out.squeeze(2)  # Shape: (batch_size_sat, batch_size_grd, w)
        print("Squeezed output: ", out.shape)
        out = out.permute(0, 2, 1)  # Shape: (batch_size_sat, w, batch_size_grd)
        print("Permutated output: ", out.shape)


        # Orientation
        orien = torch.argmax(out, dim=1)  # Shape: (batch_size_sat, batch_size_grd)
        
        return out, orien.to(dtype=torch.int32)
    
    def torch_shape(self, x, rank):
        static_shape = list(x.size())
        dynamic_shape = list(x.shape)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape[:rank])]

    def crop_sat(self, sat_matrix, orien, grd_width):
        """
        Shifts and crops the satellite feature maps based on the orientation to align them with the ground image.

        Args:
        - sat_matrix: Satellite feature maps (batch_sat, channels, height, width)
        - orien: Orientation indices (batch_sat, batch_grd)
        - grd_width: Width of the ground image

        Returns:
        - Cropped satellite feature maps aligned with the ground image
        """
        batch_sat, batch_grd = orien.shape  # Assume orien is of shape (batch_sat, batch_grd)
        _, channels, height, width = sat_matrix.shape

        # Add a dimension and repeat the satellite matrix
        sat_matrix = sat_matrix.unsqueeze(1)  # shape: [batch_sat, 1, channels, height, width]
        sat_matrix = sat_matrix.repeat(1, batch_grd, 1, 1, 1)  # shape: [batch_sat, batch_grd, channels, height, width]

        # Permute the satellite matrix to get the desired shape
        sat_matrix = sat_matrix.permute(0, 1, 4, 3, 2)  # shape: [batch_sat, batch_grd, width, height, channels]

        orien = orien.unsqueeze(-1)  # shape: [batch_sat, batch_grd, 1]

        i = torch.arange(batch_sat, device=sat_matrix.device)
        j = torch.arange(batch_grd, device=sat_matrix.device)
        k = torch.arange(width, device=sat_matrix.device)

        # Create meshgrid
        x, y, z = torch.meshgrid(i, j, k, indexing='ij')
        z_index = (z + orien) % width

        x1 = x.reshape(-1)
        y1 = y.reshape(-1)
        z1 = z_index.reshape(-1)
        index = torch.stack([x1, y1, z1], dim=1)

        # Use advanced indexing to extract the values
        sat_matrix = sat_matrix.permute(0, 1, 4, 3, 2)  # shape: [batch_sat, batch_grd, channels, height, width]
        sat_matrix_flat = sat_matrix.reshape(-1, height, channels)  # Flatten the first three dimensions
        index_flat = (x1 * batch_grd * width + y1 * width + z1).long()
        sat = sat_matrix_flat[index_flat]  # shape: [batch_sat * batch_grd * width, height, channels]
        sat = sat.reshape(batch_sat, batch_grd, width, height, channels)

        # Extract and transpose the values
        index1 = torch.arange(grd_width, device=sat_matrix.device)
        sat_transposed = sat.permute(2, 0, 1, 3, 4)  # shape: [width, batch_sat, batch_grd, height, channels]
        sat_crop_matrix = sat_transposed[index1]  # shape: [grd_width, batch_sat, batch_grd, height, channels]
        sat_crop_matrix = sat_crop_matrix.permute(1, 2, 3, 0, 4)  # shape: [batch_sat, batch_grd, height, grd_width, channels]

        assert sat_crop_matrix.shape[3] == grd_width, "The width of the cropped aerial image is incorrect"

        return sat_crop_matrix
    
    def corr_crop_distance(self, Fs, Fg):
        corr_out, corr_orien = self.corr(Fs, Fg)
        sat_cropped = self.crop_sat(Fs, corr_orien, Fg.shape[3])
        print("Satellite cropped: ", sat_cropped.shape)

        #permute channels
        sat_cropped = sat_cropped.permute(0, 1, 4, 2, 3)
        print("Satellite cropped and permuted: ", sat_cropped.shape)
        # shape = [batch_sat, batch_grd, channel, h, grd_width]

        sat_matrix = F.normalize(sat_cropped, p=2, dim=(2, 3, 4))
        distance = 2 - 2 * torch.sum(sat_matrix * Fg.unsqueeze(0), dim=(2, 3, 4)).T # shape = [batch_grd, batch_sat]

        print(f"Satellite matrix: {sat_matrix.shape}, Distance: {distance.shape}, Correlation orientation: {corr_orien.shape}")

        return sat_matrix, distance, corr_orien
