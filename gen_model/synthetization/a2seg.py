### TRANSFORM SATELLITE IMAGES TO SEGMENTATION MAPS
# This script is used to transform satellite images to segmentation maps

from san_model.model.segmentation import segmentation
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def a2seg(folder_path=None):
    """
    This function transforms satellite images in a folder to segmentation maps.
    :param folder_path: the path of the folder containing satellite images to transform
    :return: a list of segmentation maps
    """

    if folder_path is None:
        raise ValueError("A folder path must be provided.")
    
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        raise ValueError("The provided folder path does not exist or is not a directory.")
    
    # List of the images
    img_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path) if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Found {len(img_paths)} images to process.")
    
    segmentation_maps = []
    save_dir = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/my_segmented_images"
    os.makedirs(save_dir, exist_ok=True)
    
    for img_path in tqdm(img_paths, desc="Processing images"):
        seg_map = segmentation(path_img=img_path)
        
        # Convert the segmentation map to an image using a color map to visualize properly
        seg_image = Image.fromarray((seg_map * 255 / np.max(seg_map)).astype(np.uint8))
        
        # Alternative visualization with plt.imsave
        plt.imsave(os.path.join(save_dir, os.path.basename(img_path)), seg_map, cmap='viridis')
        
        segmentation_maps.append(seg_map)
    
    print("Segmentation maps saved.")
    return segmentation_maps

