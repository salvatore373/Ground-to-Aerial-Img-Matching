import numpy as np

class Transformation:
    def __init__(self, name, aerial_size, height, width):
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



