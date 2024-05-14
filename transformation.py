import numpy as np

class Transformation:
    def __init__(self, name, aerial_size = 0, height = 0, width = 0):
        """

        :param name: ????? TODO
        :param aerial_size: Size of the aerial image
        :param height: Height of the polar transformed aerial image
        :param width: Width of the polar transformed aerial image
        """
        self.name = name
        self.aerial_size = aerial_size
        self.height = height
        self.width = width

    def sample(self, img, x, y, bounds):
        x0, x1, y0, y1 = bounds
        x = np.clip(x, x0, x1-1)
        y = np.clip(y, y0, y1-1)
        return img[y, x]


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

        jj, ii = np.meshgrid(j, i) # meshgrid

        y = self.aerial_size/2. - self.aerial_size/2./self.height*(self.height-1-ii)*np.sin(2*np.pi*jj/self.width)
        x = self.aerial_size/2. + self.aerial_size/2./self.height*(self.height-1-ii)*np.cos(2*np.pi*jj/self.width)

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

        #linear interpolation in x direction
        img_x0 = (x1-x)[...,na]*img_00 + (x-x0)[...,na]*img_10
        img_x1 = (x1-x)[...,na]*img_01 + (x-x0)[...,na]*img_11

        #linear interpolation in y direction
        img_xy = (y1-y)[...,na]*img_x0 + (y-y0)[...,na]*img_x1

        return img_xy
    
    def correlation(self, Fs, Fg):
        """
        Compute the correlation between two feature maps Fs and Fg.
        
        Args:
        - Fs: map of aerial features with dimensions (H, Ws, C)
        - Fg: map of ground features with dimensions (H, Wv, C)
        
        Returns:
        - Correlation scores with dimensions (Ws,)
        """

        H, Ws, C = Fs.shape
        _, Wv, _ = Fg.shape
        correlation_scores = np.zeros(Ws)
        
        for i in range(Ws):
            correlation_score = 0
            for c in range(C):
                for h in range(H):
                    for w in range(Wv):
                        correlation_score += Fs[h, (i + w) % Ws, c] * Fg[h, w, c]
            correlation_scores[i] = correlation_score
        
        return correlation_scores
    
    def estimate_orientation(self, correlation_scores):
        """
        Estimates the orientation between ground and aerial images based on correlation scores.
        
        Args:
        - correlation_scores: Correlation scores between feature maps
        
        Returns:
        - Estimated orientation between the images
        """
        max_index = np.argmax(correlation_scores)
        estimated_orientation = max_index * (360.0 / len(correlation_scores))  # Convert the maximum index to degrees
        return estimated_orientation

    def generate_similarity_matrix(self, image_pairs):
        """
        Generates a similarity matrix between compared images.
        
        Args:
        - image_pairs: List of image pairs (matching or non-matching)
        
        Returns:
        - Similarity matrix between the compared images
        """
        scores_orientation = []
        for pair in image_pairs:
            Fs, Fg = pair
            correlation_scores = self.correlation(Fs, Fg)
            estimated_orientation = self.estimate_orientation(correlation_scores)
            scores_orientation.append((correlation_scores, estimated_orientation))

        similarity_scores = [scores for scores, _ in scores_orientation]
        orientation = [orientation for _, orientation in scores_orientation]
        similarity_matrix = np.array(similarity_scores)
        
        return orientation, similarity_matrix
    
    def shift_crop(self, Fs, similarity_matrix, shift_amount):
        """
        Shifts and crops the aerial feature maps based on the similarity matrix to align them with the ground image.
        
        Args:
        - Fs: Aerial feature maps
        - similarity_matrix: Similarity matrix between the compared images
        - shift_amount: Amount of shift in pixels
        
        Returns:
        - Shifted and cropped aerial feature maps
        """

        # Calculate the mean similarity for each column of the similarity matrix
        mean_similarity = np.mean(similarity_matrix, axis=0)
        
        # Apply a weighted shift based on the mean similarity
        weighted_shift_amount = shift_amount * mean_similarity
        
        # Apply the shift to the aerial feature maps
        shifted_Fs = np.roll(Fs, weighted_shift_amount, axis=1)  # Shift along the width dimension
        
        # Crop the aerial feature maps to match the size of the ground image
        cropped_Fs = shifted_Fs[:, :Fs.shape[1] - weighted_shift_amount, :]  # Crop the shifted feature maps
        
        return cropped_Fs



