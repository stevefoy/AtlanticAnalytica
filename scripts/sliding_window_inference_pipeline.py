from argparse import ArgumentParser
import pandas as pd
from urllib.request import urlopen
from PIL import Image,  ImageDraw, ImageFont
import timm
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import csv
import os
from typing import List

import os
from torchsummary import summary

# Workaround for duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_weights_loaded(model, args):
    """
    Checks if the weights are loaded correctly into the model by comparing
    selected model parameters before and after loading the checkpoint.

    Parameters:
    - model: The model instance into which the weights will be loaded.
    - checkpoint_path: Path to the checkpoint file.
    """
    cid_to_spid = load_class_mapping(args.class_mapping)
    spid_to_sp = load_species_mapping(args.species_mapping)


    # Function to extract and print relevant parameter snapshots
    def print_param_snapshots(parameters, label):
        print(f"\nParameter snapshots {label}:")
        for name, param in parameters:
            print(f"{name}: {param.data.flatten()[0:5]}...")  # Print the first 5 values

    # Select a few parameters for comparison (change these as per your model structure)
    selected_params = ['head.bias', 'head.weight', 'blocks.11']  # Example parameter names

    # Save initial state of selected parameters
    initial_params = [(name, param.clone()) for name, param in model.named_parameters() if name in selected_params]

    # Print initial parameter values
    print_param_snapshots(initial_params, "before loading")

    # Load the checkpoint
    #checkpoint = torch.load(checkpoint_path=args.pretrained_path, map_location='cpu')
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    
    model_state_dict = checkpoint['state_dict']
    # Print all the model
    for k, v in model_state_dict.items():
        print("", k)
        
    # Adjust keys if necessary
    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)

    # Save state of selected parameters after loading checkpoint
    loaded_params = [(name, param) for name, param in model.named_parameters() if name in selected_params]

    # Print parameter values after loading the checkpoint
    print_param_snapshots(loaded_params, "after loading")

    # Compare the snapshots
    for ((initial_name, initial_param), (loaded_name, loaded_param)) in zip(initial_params, loaded_params):
        if not torch.equal(initial_param, loaded_param):
            print(f"Parameter '{loaded_name}' has changed.")
        else:
            print(f"Warning: Parameter '{loaded_name}' unchanged. Check if this is expected.")


def modeWeightCheck(args):



    cid_to_spid = load_class_mapping(args.class_mapping)
    #--------------------------------------------------------------------------------------

    # model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid))
    XT = "D:\\pretrained_models\\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier\\model_best.pth.tar"

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=XT)

       
    check_weights_loaded(model, args)
            
     



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
    cropsloc = []
    # Adjust start and end points for both x and y coordinates to account for the border offset
    for y in range(border_offset, H - window_size - border_offset + 1, step_size):
        for x in range(border_offset, W - window_size - border_offset + 1, step_size):
            # Extract the crop
            crop = image_tensor[:, y:y+window_size, x:x+window_size]
            crops.append(crop)
            
            
    return crops


def sliding_window_batchV1(image_batch, window_size, step_size, border_offset):
    batch_crops = []
    for image_tensor in image_batch:
        # Assuming image_tensor is [C, H, W]
        C, H, W = image_tensor.shape
    # Adjust start and end points for both x and y coordinates to account for the border offset
    for y in range(border_offset, H - window_size - border_offset + 1, step_size):
        for x in range(border_offset, W - window_size - border_offset + 1, step_size):
            # Extract the crop
            crop = image_tensor[:, y:y+window_size, x:x+window_size]
            batch_crops.append(crop)

    return batch_crops

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

# Function to create a mosaic image from crops
def create_mosaic(crops, nrow):
    # Assuming crops is a list of tensors of the same shape (C, H, W)
    # Use torchvision's make_grid to create a grid of images
    grid_img = make_grid(crops, nrow=nrow)
    return grid_img

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

class ImageCropDatasetGreen(Dataset):
    def __init__(self, crops, crops_centre):
        """
        crops: List of image crops (as PIL Images or paths to images)
        """
        self.crops = crops
        self.crops_centre = crops_centre
        self.transform = T.Compose([
            # T.ToTensor(), # Convert images to tensor before normalization
            # T.Resize(size=518, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            # T.CenterCrop(size=(518, 518)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        centre = self.crops_centre[idx]
        """
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
        
        """
        green_percentage = 0
        brown_percentage = 0
        grey_percentage  = 0
        # Apply transforms to the original RGB crop
        x = self.transform(crop)
        
        # Assuming you want to return the green percentage along with the image tensor
        return x, centre, green_percentage, brown_percentage, grey_percentage

class ImageCropDataset(Dataset):
    def __init__(self, crops, transforms):
        """
        crops: List of image crops (as PIL Images or tensors)
        """
        self.crops = crops
        self.transform = T.Compose([
            # Resize only if we need to slow
            # T.Resize((518, 518)),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])

    def __len__(self):
        return len(self.crops)



    def __getitem__(self, idx):
        crop = self.crops[idx]
        x = self.transform(crop)
        return x

# Example usage:
# Assuming `crops` is a list of PIL Images or already transformed tensors
# The normalization parameters you used
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# Assuming 'normalized_image' is your normalized image tensor
# Make sure 'mean' and 'std' are reshaped to match the image tensor format ([C, H, W])
# This usually means unsqueezing dimensions to allow for broadcasting
mean = mean[:, None, None]
std = std[:, None, None]

def format_for_csv(data):
    formatted_list = []
    
    # Iterate through the complex nested list structure
    for entry in data:
        for result in entry:
            # Extracting tensor values and converting numpy arrays to string
            tensors = [str(tensor.item()) for tensor in result[0]]
            arrays = [np.array2string(array, separator=',').replace('\n', '').strip('[]') for array in result[1:]]
            
            # Combine the formatted parts
            formatted_result = ";".join(tensors + arrays)
            formatted_list.append(formatted_result)
    
    # Combine all formatted results separated by semicolon
    return ";".join(formatted_list)

# Assuming `crop` is a tensor representing an image crop
def annotate_crop(crop, annotation_text_list, device):
    # Convert tensor to PIL Image
    crop = crop.squeeze(0)

    denormalized_image = (crop * std) + mean 
    crop_image = T.ToPILImage()(denormalized_image.cpu()).convert("RGB")
    
    # Draw the annotation text on the image
    draw = ImageDraw.Draw(crop_image)
    
    font_size = 40

    # Load a .ttf or .otf font with the desired size
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    
    for pos, annotation_text in enumerate(annotation_text_list):
        text_position = (10, 35*pos)  # Top-left corner
        draw.text(text_position, annotation_text, fill=(255, 255, 0), font=font)
    
    # Convert back to tensor
    transform_back = T.Compose([
        T.ToTensor()
    ])
    crop_annotated = transform_back(crop_image).to(device)
    return crop_annotated




def visualize_attention_map(img, attention_map, block_index, head_index):
    # Assuming img is a PyTorch tensor of shape (C, H, W)
    # Convert to numpy and transpose to (H, W, C) for visualization
    denormalized_image = (img.squeeze(0) * std) + mean 
    
    img_np = denormalized_image.permute(1, 2, 0).numpy()
    print("attention_map.shape", attention_map.shape)

    simplified_attention  = attention_map.mean(dim=1)
    #print("attention_map.shape", avg_attention_weights.shape)
    #simplified_attention = avg_attention_weights.mean(dim=1).squeeze()
    print("attention_map.shape", simplified_attention.shape)

    # Assuming a square image and square patches for simplicity
    image_size = 518  # Original image size
    patch_size = 14  # Size of the patches
    num_patches_side = 37 # looking at layer 1 img // patch_size

    # Calculate how many elements are missing for the desired reshape
    missing_elements = 1369 - simplified_attention.size(0)

    # Pad the tensor with zeros (or another appropriate value) at the end
    simplified_attention = torch.nn.functional.pad(simplified_attention, (0, missing_elements), "constant", 0)

    # Now reshape it to the desired 37x37 matrix
    #reshaped_tensor = simplified_attention.reshape(37, 37)

    # Reshape attention to match the number of patches per side
    attention_map = simplified_attention.reshape((num_patches_side, num_patches_side))

    # Resize attention map to the original image dimensions
    from scipy.ndimage import zoom
    scale_factor = image_size / num_patches_side
    attention_weights_reshaped = zoom(attention_map.cpu().numpy(), scale_factor, order=1)  # Use bilinear interpolation (order=1)


    #plt.figure(figsize=(10, 10))
    #plt.imshow(img_np, alpha=0.9)  # Show the image
    #plt.imshow(attention_weights_reshaped, cmap='hot', alpha=0.5)  # Overlay the attention heatmap
    #plt.axis('off')
    #plt.show()

    # Assuming img_np is your original image and attn_map_resized is your attention map
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Creates 2 subplots side by side

    # Plot original image
    ax[0].imshow(img_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # Hide axis for better visualization

    # Plot image with attention overlay
    #ax[1].imshow(img_np)  # First, show the original image
    # Then, overlay the attention map
    ax[1].imshow(attention_weights_reshaped, cmap='jet', alpha=0.9)  # Adjust alpha to your liking
    ax[1].set_title('Image with Attention Overlay')
    ax[1].axis('off')  # Hide axis

    # Optional: Add a colorbar to indicate the scale of the attention map
    # Create an axes for colorbar. It's positioned [left, bottom, width, height] in the figure coordinate system
    cax = fig.add_axes([ax[1].get_position().x1+0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    # Create colorbar
    plt.colorbar(ax[1].images[-1], cax=cax, orientation='vertical')
    plt.suptitle('Comparison of Original and Attention Overlay')  # Optional: Add a main title
    
    plt.savefig('./crops_images/attention_map.png', bbox_inches='tight')

    plt.show(block=False)  # Display the plot without blocking the continuation of the script

    plt.pause(4)  # Pause for 0.5 seconds with the plot shown

    plt.close('all')  # Close the plot programmatically

def visualize_attention_map_old(img, attention_map, block_index, head_index):
    # Assuming img is a PyTorch tensor of shape (C, H, W)
    # Convert to numpy and transpose to (H, W, C) for visualization
    denormalized_image = (img.squeeze(0) * std) + mean 
    
    img_np = denormalized_image.permute(1, 2, 0).numpy()
    #img_np = np.resize(img_np, (img_np.shape[0], img_np.shape[1]))
    # denormalized_image = (img * std) + mean
    
    # Select attention map from a specific block and head
    # Adjust indices as necessary


    attention_map_softmax = torch.nn.functional.softmax(attention_map[block_index][head_index], dim=-1)
    attention_weights = attention_map_softmax.cpu().numpy()


    # Resize attention map to match img size (assuming square image for simplicity)
    attn_map_resized = np.resize(attention_weights, (img_np.shape[0], img_np.shape[1]))
    
    # Overlay the attention map on the image

    #plt.imshow(img_np)
    #plt.imshow(attn_map_resized, cmap='jet', alpha=0.1)  # Adjust alpha for transparency
    #plt.colorbar()
    #plt.show()

    # Assuming img_np is your original image and attn_map_resized is your attention map
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Creates 2 subplots side by side

    # Plot original image
    ax[0].imshow(img_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # Hide axis for better visualization

    # Plot image with attention overlay
    #ax[1].imshow(img_np)  # First, show the original image
    # Then, overlay the attention map
    ax[1].imshow(attn_map_resized, cmap='jet', alpha=0.4)  # Adjust alpha to your liking
    ax[1].set_title('Image with Attention Overlay')
    ax[1].axis('off')  # Hide axis

    # Optional: Add a colorbar to indicate the scale of the attention map
    # Create an axes for colorbar. It's positioned [left, bottom, width, height] in the figure coordinate system
    cax = fig.add_axes([ax[1].get_position().x1+0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    # Create colorbar
    plt.colorbar(ax[1].images[-1], cax=cax, orientation='vertical')
    plt.suptitle('Comparison of Original and Attention Overlay')  # Optional: Add a main title

    plt.show()


def visualize_attention_mapV2(attention_weights):
    # Normalize attention weights for visualization
    attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
    print(attention_weights.shape)
    # Plot attention maps for each head
    num_heads = attention_weights.size(1)
    fig, axs = plt.subplots(1, num_heads, figsize=(20, 5))

    for i in range(num_heads):
        axs[i].imshow(attention_weights[0, i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Head {i+1}')

    plt.show()

# Visualize the attention maps
#visualize_attention_map(attention_weights)

def main(args, DEBUG=True):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    image_files = find_images(args.testfolder_path)

    dataset = SpeciesImageDataset(image_files)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # Setup torch 
    device = torch.device(args.device)

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=args.pretrained_path)
    model = model.to(device)
    model = model.eval()


    # Print the summary
    # You need to specify the input size (channels, height, width)
    # For example, for a model that takes 224x224 RGB images:
    # summary(model, input_size=(3, 518, 518))
    


    attention_maps = []
    """
    def hook_function(module, input, output):
        # Assuming output is a tuple where the attention weights are the second element
        # print(output.shape)
        attention_map = output[0]  # Adjust this index based on your model's output
        attention_maps.append(attention_map.detach())
   
    
    
    model.blocks[11].attn.register_forward_hook(hook_function)
    
    """
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
    transforms_trained = timm.data.create_transform(**data_config, is_training=False)
    print("transforms_trained", transforms_trained)



    print("plot_id;species_ids")
    file_out = open("result.csv", 'w')
    file_out_prob = open("result_prob.csv", mode='w')
    
    
    file_out.write("plot_id;species_ids"+str('\n') )
    file_out_prob.write("plot_id;species_ids"+str('\n') )
    counter = 0
    
    dynamic_threshold = 15
    for img_tensor, file_path in tqdm(data_loader): 
        # print("Found image file: ", file_path)
        file_name_with_extension = os.path.basename(file_path[0])
        file_name, _ = os.path.splitext(file_name_with_extension)


        # Example usage
        window_size =  int(518)  # The size of the window
        step_size = int(518//3)    # How much the window slides each time. This could be less than window_size if you want overlapping windows
        border_offset = 150  # Starting the window 100 pixels from the border

        # Assuming image_tensor is your loaded image as a tensor
        crops,crops_centre  = sliding_window_batch(img_tensor, window_size, step_size, border_offset)

        # dataset_crops = ImageCropDataset(crops, transforms_trained)
        dataset_crops = ImageCropDatasetGreen(crops,crops_centre)
        data_crop_loader = DataLoader(dataset_crops, batch_size=1, shuffle=False, num_workers=8)

        species_id_set = set()
        species_id_max_proba = {} 
        
        top_probabilities = []
        crops_annotated = []
        for i, (crop, crop_cord, green_percentage, brown_percentage, grey_percentage) in enumerate(data_crop_loader):
            
            if True:
            #if green_percentage > 30 and grey_percentage < 25 and  brown_percentage < 10 : # if we have more 10 % in the patch then do classification
                
                img = crop.to(device)
                # print(img.shape)
                
                attention_maps.clear()
                output = model(img)  # unsqueeze single image into batch of 1
                
                # visualize_attention_mapV2(attention_maps[0])
    
                
                
                
                
                top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
                top5_probabilities = top5_probabilities.cpu().detach().numpy()
                top5_class_indices = top5_class_indices.cpu().detach().numpy()
                
                max_proba = np.max(top5_probabilities)
                data_packet = str(crop_cord[0].item()) + str(", ") + str(crop_cord[1].item()) + str(", ") + str(top5_probabilities.tolist()) 
                data_packet += str(", ") + str(top5_class_indices.tolist())
                
    
                 
                annotation_text_list =[]
                for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
                    species_id = cid_to_spid[cid]
                    # print("species_id type", type(species_id))
                    if proba > dynamic_threshold:
                        
    
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
                    
                    annotation_text = f"ID: {species_id}, P: {proba:.0f}%"
                    annotation_text_list.append(annotation_text)
                
                annotation_text_list.append(str(green_percentage.numpy()))
                annotation_text_list.append(str(brown_percentage.numpy()))
                annotation_text_list.append(str(grey_percentage.numpy()))
                    
                if DEBUG == True:
                    crop_annotated = annotate_crop( crop, annotation_text_list, device)
                    crops_annotated.append(crop_annotated)
            else:
                data_packet = str(crop_cord[0].item())+str(", ")+str(crop_cord[1].item())+str(", ")
            
            str_result_prob  = file_name.rstrip()+";"+str(i)+";"+data_packet+str("\n")
            file_out_prob.write(str_result_prob)
                
                
        
        if DEBUG == True:
            # visualize_attention_map(crop, attention_maps[0], 0, 0)
            counter = counter+1
            if len(crops_annotated) > 0:
                if len(crops_annotated)>5:
                    grid_img = make_grid(crops_annotated, nrow=5)  # Adjust `nrow` based on your preference
                else:
                    grid_img = make_grid(crops_annotated, nrow=1)
    
                # Convert grid to a PIL Image for saving
                grid_img_pil = T.ToPILImage()(grid_img.cpu()).convert("RGB")
    
                # Save or display the mosaic image
                mosaic_path = str('./crops_images/')+file_name+'mosaic_image.jpg'
    
                grid_img_pil.save(mosaic_path)
            else:
                print("error", file_name.rstrip())
            # mosaic_image.show()
        if counter > 40:
            break
        
        #dynamic_threshold = np.percentile(top5_probabilities , 90)


        str_result = file_name.rstrip()+";"+str(list(species_id_set))+str("")+str('\n') 
        
        #print(species_id_max_proba)
        
        #top_5 = np.sort(top_probabilities )[::-1][:5]


        file_out.write(str_result)

    file_out.close()
    file_out_prob.close()
    print("dynamic threshold", dynamic_threshold)
    
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


    parser.add_argument("--class_mapping", type=str, default='class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='species_id_to_name.txt')
    
    parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar')

    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\' )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)
