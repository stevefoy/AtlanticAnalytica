from argparse import ArgumentParser
import pandas as pd
from urllib.request import urlopen
from PIL import Image
import timm
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid

import numpy as np
from tqdm import tqdm

import csv
import os
from typing import List

class SpeciesData:
    def __init__(self):
        self.data = []

    def add_entry(self, plot_id, species_ids):
        """
        Add a new entry with plot ID and a list of species IDs.
        
        Args:
        - plot_id (str): The plot identifier.
        - species_ids (list of int): List of species identifiers.
        """
        self.data.append((plot_id, species_ids))

    def write_csv(self, filename):
        """
        Write the data to a CSV file with the specified format: plot_id;species_ids
        
        Args:
        - filename (str): The name of the file to write to.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for plot_id, species_ids in self.data:
                # Convert the list of species_ids to a string representation
                species_ids_str = "[" + ",".join(map(str, species_ids)) + "]"
                writer.writerow([plot_id, species_ids_str])


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

def sliding_window(image_tensor, window_size=518, step_size=518, border_offset=100):
    # C, H, W are the channel, height, and width of the image tensor
    C, H, W = image_tensor.shape
    crops = []

    # Adjust start and end points for both x and y coordinates to account for the border offset
    for y in range(border_offset, H - window_size - border_offset + 1, step_size):
        for x in range(border_offset, W - window_size - border_offset + 1, step_size):
            # Extract the crop
            crop = image_tensor[:, y:y+window_size, x:x+window_size]
            crops.append(crop)
            
    return crops

# Function to create a mosaic image from crops
def create_mosaic(crops, nrow):
    # Assuming crops is a list of tensors of the same shape (C, H, W)
    # Use torchvision's make_grid to create a grid of images
    grid_img = make_grid(crops, nrow=nrow)
    return grid_img


def main(args):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    spid_to_sp = load_species_mapping(args.species_mapping)
    image_files = find_images(args.testfolder_path)
    
    # Setup torch 
    device = torch.device(args.device)

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=args.pretrained_path)
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
    transforms = timm.data.create_transform(**data_config, is_training=False)



    # DEBUG CODE I image
    if False:
        # image_files = list([image_files[0]])
        image_files = image_files[0:10]

    print("plot_id;species_ids")
    file_out = open("result.csv", 'w')
    file_out_prob = open("result_prob.csv", mode='w')
    
    
    file_out.write("plot_id;species_ids"+str('\n') )
    file_out_prob.write("plot_id;species_ids"+str('\n') )
    for file_path in tqdm(image_files):
        # print("Found image file: ", file_path)
        file_name_with_extension = os.path.basename(file_path)

        # Split the name and extension
        file_name, _ = os.path.splitext(file_name_with_extension)
        # print("fullpath: ", file_path)
        # print("filename: ", file_name)

        image = Image.open(file_path)


        # Transform the image into a PyTorch tensor
        transform_to_tensor = T.ToTensor()
        img_tensor = transform_to_tensor(image)
        # Example usage
        window_size = 518  # The size of the window
        step_size = 518    # How much the window slides each time. This could be less than window_size if you want overlapping windows
        border_offset = 150  # Starting the window 100 pixels from the border

        # Assuming image_tensor is your loaded image as a tensor
        crops = sliding_window(img_tensor, window_size, step_size, border_offset)

        if True:
            # If you have a list of crops and want to save or process them, you can iterate through `crops`.
            # Example usage
            nrow = 5  # Number of images per row in the mosaic
            # Assuming crops is a list of tensors from your sliding window function
            grid_img = create_mosaic(crops, nrow)

            # Convert the grid to a PIL Image for saving or displaying
            mosaic_image =  T.ToPILImage()(grid_img)
            # Save or display the mosaic image
            mosaic_path = str('./crops_images/')+file_name+'mosaic_image.jpg'
            mosaic_image.save(mosaic_path)
            # mosaic_image.show()


        species_id_set = set()
        species_id_max_proba = {} 
        top_probabilities = []

        for i, crop in enumerate(crops):
            crop_image =  T.ToPILImage()(crop)
            img = transforms(crop_image).unsqueeze(0)
            img = img.to(device)
            # print(img.shape)
            output = model(img)  # unsqueeze single image into batch of 1
            top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
            top5_probabilities = top5_probabilities.cpu().detach().numpy()
            top5_class_indices = top5_class_indices.cpu().detach().numpy()
            
            max_proba = np.max(top5_probabilities)
            top_probabilities.append(max_proba)

             

            for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
                species_id = cid_to_spid[cid]
                # print("species_id type", type(species_id))
                if proba > 50:
                    species_id_set.add(int(species_id))
                    species = spid_to_sp[species_id]
                    # print(species_id, species, proba)
                    # Check if this species_id already has a recorded probability
                    if species_id in species_id_max_proba:
                        # If the current probability is higher, update it
                        if proba > species_id_max_proba[species_id]:
                            species_id_max_proba[species_id] = proba
                    else:
                        # If the species_id is not in the dictionary, add it
                        species_id_max_proba[species_id] = proba
        
        dynamic_threshold = np.percentile(top5_probabilities , 90)

        str_result = file_name.rstrip()+";"+str(list(species_id_set))+str(";")+str('\n') 
        # print(dynamic_threshold)
        #print(species_id_max_proba)
        
        top_5 = np.sort(top_probabilities )[::-1][:5]

        str_result_top_5  = file_name.rstrip()+";"+str(dynamic_threshold)+str(';')+str(list(top_5))+str(";")+str('\n') 
        file_out_prob.write(str_result_top_5)
        file_out.write(str_result)

    file_out.close()
    file_out_prob.close()
    
    """  


    img = None
    if 'https://' in args.image or 'http://' in  args.image:
        img = Image.open(urlopen(args.image))
    elif args.image != None:
        img = Image.open(args.image)
        
    if img != None:
        img = transforms(img).unsqueeze(0)
        img = img.to(device)
        # print(img.shape)
        output = model(img)  # unsqueeze single image into batch of 1
        top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
        top5_probabilities = top5_probabilities.cpu().detach().numpy()
        top5_class_indices = top5_class_indices.cpu().detach().numpy()

        for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
            species_id = cid_to_spid[cid]
            species = spid_to_sp[species_id]
            print(species_id, species, proba)

        species_data = SpeciesData()
        species_data.add_entry("CBN-Pla-B1-20130724", [1395806])
        species_data.add_entry("CBN-PdlC-A1-20130807", [1351284, 1494911, 1381367, 1396535, 1412857, 1295807])
    
    # Write to CSV file
    species_data.write_csv("species_data.csv")
    """ 


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image", type=str, default='') #Orchis simia

    parser.add_argument("--class_mapping", type=str) #'class_mapping.txt'
    parser.add_argument("--species_mapping", type=str) #'species_id_to_name.txt'
    
    parser.add_argument("--pretrained_path", type=str) #model_best.pth.tar

    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--testfolder_path", type=str)
    
    args = parser.parse_args()
    main(args)
