# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:50:53 2024

@author: stevf
"""
from argparse import ArgumentParser
import pandas as pd
import timm
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image,  ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import functional as TF
import numpy as np

class SpeciesImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.transform = T.Compose([
            T.ToTensor(),  # Converts PIL.Image.Image to torch.FloatTensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, image_path
    


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name

def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

def find_images(directory):
    # List of image file extensions you're interested in
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    image_files.sort()

    return image_files

def sliding_window_batch(image_batch, window_size, step_size, border_offset):
    batch_crops = []
    crop_centers = []  # List to hold the center coordinates of each crop

    for image_tensor in image_batch:
        # Assuming image_tensor is [C, H, W]
        C, H, W = image_tensor.shape
        # Adjust start and end points for both x and y coordinates to account for the border offset
        for y in range(border_offset, H - window_size - border_offset + 1, step_size):
            for x in range(border_offset, W - window_size - border_offset + 1, step_size):
                # Extract the crop
                crop = image_tensor[:, y:y+window_size, x:x+window_size]
                batch_crops.append(crop)
                # Calculate the center coordinates of the current crop
                center_x = x + window_size // 2
                center_y = y + window_size // 2
                
                crop_centers.append([center_x, center_y])

    return batch_crops, crop_centers

class ImageCropDatasetGreen(Dataset):
    def __init__(self, crops, crops_centre):
        """
        crops: List of image crops (as PIL Images or paths to images)
        """
        self.crops = crops
        self.crops_centre = crops_centre
        self.transform = T.Compose([
            # T.ToTensor(), # Convert images to tensor before normalization
            T.Resize(size=518, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(518, 518)),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        centre = self.crops_centre[idx]
        # Convert tensor to PIL Image in RGB
        img_pil_rgb = to_pil_image(crop)
        
        # Convert PIL Image from RGB to HSV
        img_pil_hsv = img_pil_rgb.convert('HSV')
        
        
        np_hsv_crop = np.array(img_pil_hsv)


        # Define your green range in HSV
        #lower_green = np.array([35, 100, 100])
        #upper_green = np.array([75, 255, 255])
        
        lower_green = np.array([30, 50, 50])  # loking at gimp more yellow-greens and less saturated greens
        upper_green = np.array([85, 255, 255]) # Extend to bluish-greens and very bright greens

        # Mask to detect green areas
        green_mask = np.all(np_hsv_crop >= lower_green, axis=-1) & np.all(np_hsv_crop <= upper_green, axis=-1)
        green_percentage = int(np.mean(green_mask) * 100)


        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 150, 150])
        
        
        # Mask to detect green areas
        brown_mask = np.all(np_hsv_crop >= lower_brown, axis=-1) & np.all(np_hsv_crop <= upper_brown, axis=-1)
        brown_percentage = int(np.mean(brown_mask) * 100)


        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 150, 150])
        
        
        # Mask to detect green areas
        brown_mask = np.all(np_hsv_crop >= lower_brown, axis=-1) & np.all(np_hsv_crop <= upper_brown, axis=-1)
        brown_percentage = int(np.mean(brown_mask) * 100)
        
        
        lower_grey = np.array([0, 0, 50])  # Assuming a bit of leeway for 'dark greys'
        upper_grey = np.array([180, 50, 200])  # Broad hue range, low saturation, moderate-high value

        # Mask to detect green areas
        grey_mask = np.all(np_hsv_crop >= lower_grey, axis=-1) & np.all(np_hsv_crop <= upper_grey, axis=-1)
        grey_percentage = int(np.mean(grey_mask) * 100)
        

        # Apply transforms to the original RGB crop
        x = self.transform(crop)
        
        # Assuming you want to return the green percentage along with the image tensor
        return x, centre, green_percentage, brown_percentage, grey_percentage

def main(args, DEBUG=True):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)

    image_files = find_images(args.testfolder_path)
    dataset = SpeciesImageDataset(image_files)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)


    



    print("plot_id;species_ids")
    file_out = open("resultx.csv", 'w')
    file_out_prob = open("result_probx.csv", mode='w')
    
    
    file_out.write("plot_id;species_ids"+str('\n') )
    file_out_prob.write("plot_id;species_ids"+str('\n') )
    counter = 0
    
    dynamic_threshold = 20
    for img_tensor, file_path in tqdm(data_loader): 
        # print("Found image file: ", file_path)
        file_name_with_extension = os.path.basename(file_path[0])

        parent_path = os.path.dirname(file_path[0])
        # Drop back from image folder one level to save
        back_one_directory = os.path.dirname(parent_path)
        
        file_name, _ = os.path.splitext(file_name_with_extension)
        
        
        new_folder = os.path.join(back_one_directory, file_name)
        # print(" parent_folder",  parent_path)

        # Example usage
        window_size =  int(518)  # The size of the window
        step_size = int(518)    # How much the window slides each time. This could be less than window_size if you want overlapping windows
        border_offset = 0  # Starting the window 100 pixels from the border

        # Assuming image_tensor is your loaded image as a tensor
        crops, crops_centre  =  sliding_window_batch(img_tensor, window_size, step_size, border_offset)
        
        
        # dataset_crops = ImageCropDataset(crops, transforms_trained)
        dataset_crops = ImageCropDatasetGreen(crops,crops_centre)
        data_crop_loader = DataLoader(dataset_crops, batch_size=1, shuffle=False, num_workers=8)

        species_id_set = set()
        species_id_max_proba = {} 
        
        top_probabilities = []
        crops_annotated = []
        
        if not os.path.exists(new_folder):
            # If it doesn't exist, create it
            # pass
            os.makedirs(new_folder)
        
        save_file_path = os.path.join(new_folder, 'prob.csv')
        file_out_prob = open(save_file_path, mode='w')
        
        for i, (crop, crop_cord, green_percentage, brown_percentage, grey_percentage) in enumerate(data_crop_loader):
            
            # torch.save(crop, 'cropped_image.pt')
            crop_image = TF.to_pil_image(crop.squeeze(0))
            # print(type(crop), crop.shape)
            save_file_path = os.path.join(new_folder, str(i)+'idx_'+file_name+'_image.png')
            crop_image.save(save_file_path)
            
            data_packet = str(crop_cord[0].item()) + str(", ") + str(crop_cord[1].item()) + str(", ") + str(green_percentage.item() )
            data_packet +=  str(", ") + str(brown_percentage.item()) + str(", ") + str(grey_percentage.item()) 
            str_result_prob  = file_name.rstrip()+";"+str(i)+";"+data_packet+str("\n")
            file_out_prob.write(str_result_prob)
            
        file_out_prob.close()    
        
        
if __name__ == '__main__':
    parser = ArgumentParser()


    parser.add_argument("--class_mapping", type=str, default='D:\\PlantCLEF2024\\class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='D:\\PlantCLEF2024\\species_id_to_name.txt')
    
    parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar')

    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\' )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)