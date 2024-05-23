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
from torchvision.datasets import ImageFolder


import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import csv
import os

from torch.cuda.amp import autocast, GradScaler


import os
from torchsummary import summary

def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

class Clef2024Dataset(Dataset):
    def __init__(self, data_dir, class_mapping_file, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Read class mapping
        with open(class_mapping_file, 'r') as f:
            class_order = [line.strip() for line in f]
        
        # Create a mapping from folder name to a continuous index
        folder_to_idx = {folder_name: i for i, folder_name in enumerate(class_order)}
        
        # Traverse through the data directory
        for folder_name in class_order:
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):  # Ensure the folder exists
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                        file_path = os.path.join(folder_path, file_name)
                        self.data.append(file_path)
                        self.labels.append(folder_to_idx[folder_name])
                        #print(folder_path," map",folder_to_idx[folder_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        # Apply transformation if it's set
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx] #,img_path





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


class loadImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.transform= T.Compose([
            T.ToTensor(),  # Converts PIL.Image.Image to torch.FloatTensor
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if image.size == (0, 0):  # Check if image dimensions are zero
            raise ValueError(f"Image at index {index} is empty with path {image_path}")
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
            T.Resize(size=518, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(518, 518)),
            #T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        centre = self.crops_centre[idx]

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


import torch.optim as optim
import torch.nn as nn


def main(args, DEBUG=True):
    
    
    # Transformation setup
    transform_clef = T.Compose([
        T.Resize((518, 518)),
        T.CenterCrop(size=(518, 518))
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset instantiation
    dataset = Clef2024Dataset(data_dir='D:\\PlantCLEF2024\\PlantCLEF2024\\train',
                              class_mapping_file='D:\\PlantCLEF2024\\class_mapping.txt', 
                              transform=transform_clef)





    cid_to_spid = load_class_mapping('class_mapping.txt')
    spid_to_sp = load_species_mapping('species_id_to_name.txt')

    #print("Label for index 16:", cid_to_spid[label], spid_to_sp[cid_to_spid[label]])


   #raise ValueError("bad")
    
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    
    

      
    


    # Setup torch 
    device = torch.device(args.device)


    
    #   15947MiB was 16, 
    data_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=8)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=args.pretrained_path)
    model = model.to(device)
    
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
    transforms_trained = timm.data.create_transform(**data_config, is_training=False)
    print("transforms_trained", transforms_trained)
    
    
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()  # Set model to training mode
    num_epochs = 20
    
    for epoch in range(num_epochs):
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



if __name__ == '__main__':
    parser = ArgumentParser()


    parser.add_argument("--class_mapping", type=str, default='class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='species_id_to_name.txt')
    
    parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar')
    # parser.add_argument("--pretrained_path", type=str, default='./vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier/model_best.pth.tar')
    
    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\' )
   # parser.add_argument("--testfolder_path", type=str, default='D:\\PlantCLEF2024\\annotated\\images\\' )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)
