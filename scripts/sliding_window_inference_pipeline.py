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
import psutil
import cv2

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


def DEL_modeWeightCheck(args):



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




def sliding_window_batch(image_batch, file_nameCSV, window_size, step_size, border_offset):
    """
    Generates image crops using a sliding window approach along with their coordinates.

    Parameters:
    - image_batch: a batch of images in tensor format [N, C, H, W]
    - file_nameCSV: the base file name for output
    - window_size: the size of each window crop
    - step_size: the step size for the sliding window
    - border_offset: offset from the image border to start cropping

    Returns:
    - batch_crops: a list of crops from the image batch
    - crop_coordinates: a list of strings with filename and crop coordinates
    """
    batch_crops = []
    crop_coordinates = []

    for image_tensor in image_batch:
        # Assuming image_tensor is [C, H, W]
        C, H, W = image_tensor.shape
        # Adjust start and end points for both x and y coordinates to account for the border offset
        for y in range(border_offset, H - window_size - border_offset + 1, step_size):
            for x in range(border_offset, W - window_size - border_offset + 1, step_size):
                # Extract the crop
                crop = image_tensor[:, y:y+window_size, x:x+window_size]
                batch_crops.append(crop)

                # Generate the filename and coordinates string
                # This string includes the filename and the x, y coordinates for the top-left and bottom-right corners of the crop
                coordinates = f"{file_nameCSV},{y},{y + window_size},{x},{x + window_size}"
                crop_coordinates.append(coordinates)

    return batch_crops, crop_coordinates

def sliding_window_batchWH(image_batch, file_nameCSV, window_height, window_width, step_size, border_offset):
    """
    Generates image crops using a sliding window approach along with their coordinates,
    allowing for different window heights and widths.

    Parameters:
    - image_batch: a batch of images in tensor format [N, C, H, W]
    - file_nameCSV: the base file name for output
    - window_height: the height of each window crop
    - window_width: the width of each window crop
    - step_size: the step size for the sliding window
    - border_offset: offset from the image border to start cropping

    Returns:
    - batch_crops: a list of crops from the image batch
    - crop_coordinates: a list of strings with filename and crop coordinates
    """
    if window_height <= 0 or window_width <= 0 or step_size <= 0:
        raise ValueError("window_height, window_width, and step_size must be positive integers")
    
    batch_crops = []
    crop_coordinates = []

    for idx, image_tensor in enumerate(image_batch):
        C, H, W = image_tensor.shape  # Image dimensions: channels, height, width
        # Ensure the window fits within the image dimensions adjusted by the border offset
        if H < window_height + 2 * border_offset or W < window_width + 2 * border_offset:
            continue

        start_range_h = border_offset
        end_range_h = H - window_height - border_offset + 1

        start_range_w = border_offset
        end_range_w = W - window_width - border_offset + 1

        for y in range(start_range_h, end_range_h, step_size):
            for x in range(start_range_w, end_range_w, step_size):
                crop = image_tensor[:, y:y + window_height, x:x + window_width]
                batch_crops.append(crop)

                # Include the image index and exact coordinates in the filename
                coordinates = f"{file_nameCSV}_img{idx}_{y}_{y + window_height}_{x}_{x + window_width}"
                crop_coordinates.append(coordinates)

    return batch_crops, crop_coordinates



# Function to create a mosaic image from crops
def create_mosaic(crops, nrow):
    # Assuming crops is a list of tensors of the same shape (C, H, W)
    # Use torchvision's make_grid to create a grid of images
    grid_img = make_grid(crops, nrow=nrow)
    return grid_img

# Limited speed up

    
class loadImageDatasetOLD(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if image.size == (0, 0):  # Check if image dimensions are zero
            raise ValueError(f"Image at file index {idx} is zero with path {image_path}")


        return image, image_path

class loadImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.transform = T.Compose([
            T.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_files)



    def histogram_equalization(self, img):
        # Convert PIL image to numpy array (RGB)
        img_np = np.array(img)
        
        # Convert RGB to YCrCb
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        
        # Equalize the histogram of the Y channel (luminance)
        img_ycrcb[:,:,0] = cv2.equalizeHist(img_ycrcb[:,:,0])
        
        # Convert back to RGB
        img_eq = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        
        # Convert numpy array back to PIL Image
        img_eq_pil = Image.fromarray(img_eq)
        
        return img_eq_pil

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if image.size == (0, 0):  # Check if image dimensions are zero
            raise ValueError(f"Image at file index {idx} is zero with path {image_path}")

        # Apply histogram equalization to the image
        image_eq = self.histogram_equalization(image)
        
        # Apply transformation to tensor
        image_tensor = self.transform(image_eq)

        return image_tensor, image_path

class loadImageDatasetV2(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    def __len__(self):
        return len(self.image_files)

    def histogram_equalization(self, img_tensor):
        img_flat = img_tensor.flatten()
        bins = 256

        # Calculate histogram
        hist = torch.histc(img_flat, bins=bins, min=0, max=1)

        # Calculate CDF
        cdf = hist.cumsum(0)
        cdf_min = cdf[cdf > 0].min()
        cdf_max = cdf[-1]

        # Normalize the CDF
        cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min) * 255

        # Map the old pixel values to the new ones based on the equalized histogram
        indices = (img_flat * (bins-1)).long()
        img_eq = cdf_normalized[indices]

        # Reshape back to the original shape
        return img_eq.reshape(img_tensor.shape)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if image.size == (0, 0):  # Check if image dimensions are zero
            raise ValueError(f"Image at file index {idx} is zero with path {image_path}")

        image_tensor = self.to_tensor(image)

        # Apply histogram equalization to each channel separately
        equalized_channels = [self.histogram_equalization(channel) for channel in image_tensor]
        image_tensor_eq = torch.stack(equalized_channels)

        return image_tensor_eq, image_path



class ImageCropDataset(Dataset):
    def __init__(self, crops, crops_centre):
        """
        crops: List of image crops (as PIL Images or paths to images)
        """
        self.crops = crops
        self.crops_centre = crops_centre
        self.transform = T.Compose([
            #T.Resize(size=518, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            #T.CenterCrop(size=(518, 518)),
            #T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        centre = self.crops_centre[idx]

        # Apply transforms to the original RGB crop
        x = self.transform(crop)
        
        # Assuming you want to return the green percentage along with the image tensor
        return x, centre



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

rotation_transforms = [
    T.Compose([
        T.RandomRotation([0, 0]),  # No rotation
    ]),
    T.Compose([
        T.RandomRotation([90, 90]),  # 90 degrees
    ]),
    T.Compose([
        T.RandomRotation([180, 180]),  # 180 degrees
    ])
]



def main(args, DEBUG=True):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    
    
    # load the image files for clef full
    file_path = r"C:\Users\stevf\OneDrive\Documents\datasets\test_set\imagelist.txt"
    image_full_path = r"C:\Users\stevf\OneDrive\Documents\datasets\test_set\images"
    
    # load the image files for clef annotation test
    file_path = r"D:\PlantCLEF2024\annotated\imagelist.txt"
    image_full_path = r"D:\PlantCLEF2024\annotated\images"
      
    from datetime import datetime

    # Get current date and time
    now = datetime.now()
    
    # Convert to a very short string format, including 24-hour and minute (e.g., "202405061530" for May 6, 2024, 3:30 PM)
    short_date_time_string = now.strftime("%m%d%H%M")
    short_date_time_string = now.strftime("%m%d%H%M")
    result_path = r"D:\PlantCLEF2024\annotated\Results"
   
    csv_file = "bb518_s112_R3_eql_T"+short_date_time_string+".csv"
    csv_file = os.path.join(result_path, csv_file)
    
    image_files = []
    with open(file_path, 'r') as file:
    # Read all lines in the file and strip newline characters
        image_files = [image_full_path+line.strip() for line in file]
    
    # image_files = find_images(args.testfolder_path)

    dataset = loadImageDataset(image_files)
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
    


    counter = 0
    

    
    csv_headers = ["filename", "x1", "y1", "x2", "y2", "crop_index", "class_index_1", "probability_1", "class_index_2", "probability_2", "class_index_3", "probability_3", "class_index_4", "probability_4", "class_index_5", "probability_5"]

    # Open the CSV file for writing
    file_out_prob = open(csv_file, mode='w', newline='')
    writer = csv.writer(file_out_prob)
    writer.writerow(csv_headers)  # Write the header
    
    dynamic_threshold = 15
    for img_tensor, file_path in tqdm(data_loader): 
        # print("Found image file: ", file_path)
        file_name_with_extension = os.path.basename(file_path[0])
        file_name, _ = os.path.splitext(file_name_with_extension)
        file_nameCSV = file_name.rstrip()

        # Example usage
        
        window_size =  int(518)  # The size of the window
        step_size = int(50)    # How much the window slides each time. This could be less than window_size if you want overlapping windows
        border_offset = 50  # Starting the window 100 pixels from the border

        # Assuming image_tensor is your loaded image as a tensor
        crops, crop_name_data  = sliding_window_batch(img_tensor, file_nameCSV, window_size, step_size, border_offset)
        #crops, crop_name_data  = sliding_window_batchWH(img_tensor, file_nameCSV, window_size, step_size, border_offset)
        # dataset_crops = ImageCropDataset(crops, transforms_trained)
        dataset_crops = ImageCropDataset(crops,crop_name_data)
        data_crop_loader = DataLoader(dataset_crops, batch_size=8, shuffle=False, num_workers=8)

        species_id_set = set()
        species_id_max_proba = {} 
        
        top_probabilities = []
        crops_annotated = []
        crop_index = 0  # Initialize a separate crop index
        for batch_idx, (crops, crop_coord_data) in enumerate(data_crop_loader):
            

            #if green_percentage > 30 and grey_percentage < 25 and  brown_percentage < 10 : # if we have more 10 % in the patch then do classification
                
            imgs = crops.to(device)
            
            #for rotation_transform in rotation_transforms:
            #    rotated_imgs = rotation_transform(imgs)  # Apply rotation and other transformations
            output = model(imgs)
            
                
            top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5, dim=1)
            
            
            for i in range(imgs.size(0)):  # Iterate through each image in the batch
                probabilities = top5_probabilities[i].cpu().detach().numpy()
                class_indices = top5_class_indices[i].cpu().detach().numpy()
                
                # Parse coordinates and file name
                fn, x, y, x2, y2 = crop_coord_data[i].split(",")
                # Assuming cid_to_spid maps class indices to species IDs
                species_ids = [cid_to_spid.get(ci, "Unknown") for ci in class_indices]
                
                # Preparing the row to write, now including the filename
                row = [fn, x, y, x2, y2, crop_index] + \
                      [item for pair in zip(species_ids, probabilities) for item in pair]
                writer.writerow(row)
    
                crop_index += 1  # Increment the crop index for the next row
                        

            


    file_out_prob.close()


if __name__ == '__main__':
    parser = ArgumentParser()


    parser.add_argument("--class_mapping", type=str, default='class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='species_id_to_name.txt')
    
    parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar')
    # parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier/model_best.pth.tar')
    
    parser.add_argument("--device", type=str, default='cuda')

   # parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\' )
    parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\annotated\\images\\' )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)
