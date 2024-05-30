import enum
import re

import pandas as pd
import torch
import torchvision as vision
from torch.utils.data import Dataset


class ImageTypes(enum.Enum):
    SyntheticSat = "synthetic_sat"
    SegmentedSat = "segmented_sat"
    PolarSat = "polar_sat"
    PolarSegmentedSat = "polar_seg_sat"
    Ground = "ground"
    Sat = "sat"


class CrossViewDataset(Dataset):
    def __init__(self,
                 csv_path,
                 base_path,
                 dataset_content=[ImageTypes.PolarSat, ImageTypes.Sat, ImageTypes.Ground],
                 # pairs: bingmap, polarmap, streetview
                 normalize_imgs: bool = False,
                 use_our_data: bool = False,
                 device: torch.device = None):
        """
        Create a CrossViewDataset with the pairs contained in csv_path.

        :param csv_path: The path to the CSV file containing the bingmap-streetview pairs
        :param base_path: The path to the dataset.
        :param dataset_content: A list containing the content of each returned triplet:
        e.g. [ImageTypes.PolarSat, ImageTypes.PolarSegmentedSat, ImageTypes.Ground]
        :param normalize_imgs: Whether to normalize the images before retutning them.
        :param use_our_data: Whether the polar and segmented images are the ones loaded from the official repo
         or the ones computed by us.
        :param device: The device where the created tensors will be stored.
        """
        self.normalize_imgs = normalize_imgs

        # channel-first params
        self.ground_mean = torch.Tensor([[0.47, 0.46, 0.48]]).view(3, 1, 1)
        self.ground_std = torch.Tensor([[0.21, 0.24, 0.20]]).view(3, 1, 1)
        self.sat_mean = torch.Tensor([[[0.2420, 0.2506, 0.2173]]]).view(3, 1, 1)
        self.sat_std = torch.Tensor([[0.2778, 0.2844, 0.2583]]).view(3, 1, 1)
        self.sat_polar_mean = torch.Tensor([[0.40, 0.36, 0.41]]).view(3, 1, 1)
        self.sat_polar_std = torch.Tensor([[0.15, 0.15, 0.14]]).view(3, 1, 1)
        self.my_sat_polar_mean = torch.Tensor([[0.6321, 0.6321, 0.6321]]).view(3, 1, 1)
        self.my_sat_polar_std = torch.Tensor([[0.6574, 0.6575, 0.6574]]).view(3, 1, 1)
        self.seg_mean = torch.Tensor([[0.33, 0.85, 0.8]]).view(3, 1, 1)
        self.seg_std = torch.Tensor([[0.38, 0.27, 0.28]]).view(3, 1, 1)
        self.my_seg_mean = torch.Tensor([[0.1747, 0.0265, 0.2109, 0.6325]]).view(4, 1, 1)
        self.my_seg_std = torch.Tensor([[0.1825, 0.0592, 0.2195, 0.6573]]).view(4, 1, 1)
        self.seg_polar_mean = torch.Tensor([[0.2315, 0.5168, 0.5536]]).view(3, 1, 1)
        self.seg_polar_std = torch.Tensor([[0.3388, 0.5792, 0.5964]]).view(3, 1, 1)

        match_folder_regex = r'^.+(?=\/)'
        replace_dict = {}
        self.normal_params_per_col = []
        for ind, img_type in enumerate(dataset_content):
            curr_cont = {}
            if img_type == ImageTypes.PolarSat:
                curr_cont[match_folder_regex] = 'my_sat_polar' if use_our_data else 'polarmap/normal'
                self.normal_params_per_col.append((self.my_sat_polar_mean, self.my_sat_polar_std) if use_our_data else (
                self.sat_polar_mean, self.sat_polar_std))
            elif img_type == ImageTypes.Sat:
                curr_cont[match_folder_regex] = 'bingmap'
                self.normal_params_per_col.append((self.sat_mean, self.sat_std))
            elif img_type == ImageTypes.Ground:
                curr_cont[match_folder_regex] = 'streetview'
                curr_cont['input'] = ''
                curr_cont['png'] = 'jpg'
                self.normal_params_per_col.append((self.ground_mean, self.ground_std))
            elif img_type == ImageTypes.SegmentedSat:
                curr_cont[match_folder_regex] = 'my_segsat' if use_our_data else 'segmap'
                curr_cont['input'] = 'output'
                self.normal_params_per_col.append(
                    (self.my_seg_mean, self.my_seg_std) if use_our_data else (self.seg_mean, self.seg_std))
            elif img_type == ImageTypes.PolarSegmentedSat:
                curr_cont[match_folder_regex] = 'my_segsat_polar'
                curr_cont['input'] = 'output'
                self.normal_params_per_col.append((self.seg_polar_mean, self.seg_polar_std))
            elif img_type == ImageTypes.SyntheticSat:
                # todo: complete below
                curr_cont[match_folder_regex] = 'my_segsat_polar'
                curr_cont['input'] = 'output'
                self.normal_params_per_col.append((self.seg_polar_mean, self.seg_polar_std))

            replace_dict[ind] = curr_cont

        self.pairs = pd.read_csv(csv_path, header=None)
        self.pairs.replace(replace_dict, regex=True, inplace=True)

        if base_path[-1] == '/':
            base_path = base_path[:-1]
        self.pairs = self.pairs.map(lambda el: f'{base_path}/{el}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx, :]
        ids = [re.search(r'/([^/]+)\..+$', path).group(1) for path in row]

        img1 = vision.io.read_image(row[0])
        img2 = vision.io.read_image(row[1])
        img3 = vision.io.read_image(row[2])

        if self.normalize_imgs:
            img1 = img1.div(255).add(self.normal_params_per_col[0][0]).div(self.normal_params_per_col[0][1])
            img2 = img2.div(255).add(self.normal_params_per_col[1][0]).div(self.normal_params_per_col[1][1])
            img3 = img3.div(255).add(self.normal_params_per_col[2][0]).div(self.normal_params_per_col[2][1])

        return img1, img2, img3, *ids
