# -*- coding: utf-8 -*-
from tqdm import tqdm
import os 

def createImageList(image_list_path, subfolder_path):
    # Read the list of image paths from the file
    image_paths =[]
    with open(image_list_path, 'r') as file:
        image_paths = file.readlines()

    # Initialize lists to store all RGB values of all images
    all_images_rgb_paths = []
    
    missing_images = []
    for image_path in tqdm(image_paths):
        image_path = image_path.strip()  # Remove any leading/trailing whitespace
        # Construct the expected path within the subfolder
        image_name = os.path.basename(image_path)
        expected_path = os.path.join(subfolder_path, image_name)
        print("found:", expected_path)

        if not os.path.exists(expected_path):
            missing_images.append(image_path)
        else:
            # Add the image's RGB values to the list
            all_images_rgb_paths.append(expected_path)
            
    # Report missing images
    if missing_images:
        print("NOT FOUND images as in 'imagelist.txt' ")
        for missing in missing_images:
            print(missing)
            
    return all_images_rgb_paths