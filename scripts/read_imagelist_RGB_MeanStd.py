import os
import argparse
from PIL import Image
import numpy as np




def validate_images(image_list_path, subfolder_path):
    """
    Validates if images listed in image_list_path exist in subfolder_path and calculates their RGB mean and standard deviation.

    Args:
    image_list_path (str): Path to the file containing the image list.
    subfolder_path (str): Path to the subfolder to check for image existence.
    """
    # Read the list of image paths from the file
    with open(image_list_path, 'r') as file:
        image_paths = file.readlines()

    missing_images = []
    for image_path in image_paths:
        image_path = image_path.strip()  # Remove any leading/trailing whitespace
        # Construct the expected path within the subfolder
        image_name = os.path.basename(image_path)
        expected_path = os.path.join(subfolder_path, image_name)

        if not os.path.exists(expected_path):
            missing_images.append(image_path)
        else:
            # Calculate and print the RGB mean and standard deviation for the image
            calculate_image_stats(expected_path)

    # Report missing images
    if missing_images:
        print("The following images are listed in 'imagelist.txt' but were not found in the specified subfolder:")
        for missing in missing_images:
            print(missing)
    else:
        print("All images listed in 'imagelist.txt' were found and analyzed.")

def calculate_image_stats(image_path):
    """
    Calculates and prints the RGB mean and standard deviation for an image.

    Args:
    image_path (str): Path to the image file.
    """
    # Open the image
    image = Image.open(image_path)
    # Convert the image into a numpy array and separate the channels
    image_array = np.array(image)
    # Check if the image is grayscale; if so, replicate the values across three channels for consistency
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] > 3:
        # If the image has an alpha channel, ignore it
        image_array = image_array[:, :, :3]

    # Calculate mean and standard deviation
    mean = np.mean(image_array, axis=(0, 1))
    std = np.std(image_array, axis=(0, 1))

    print(f"Image: {os.path.basename(image_path)} - RGB Mean: {mean}, RGB Standard Deviation: {std}")




if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate existence of images and calculate their RGB statistics.")
    parser.add_argument("--image_list_path", type=str, help="Path to the file containing the image list.")
    parser.add_argument("--subfolder_path", type=str, help="Path to the subfolder to check for image existence.")

    args = parser.parse_args()

    # Perform the validation and calculations
    validate_images(args.image_list_path, args.subfolder_path)
    
    
    
    
    
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv  # Import csv module

def validate_images_and_calculate_stats(image_list_path, subfolder_path, output_csv_path):
    """
    Validates if images listed in image_list_path exist in subfolder_path, calculates their RGB mean and standard deviation,
    aggregates this information to calculate the overall mean and STD for the folder, and saves the results to a CSV file.
    
    Args:
    image_list_path (str): Path to the file containing the image list.
    subfolder_path (str): Path to the subfolder to check for image existence.
    output_csv_path (str): Path to the output CSV file where results will be saved.
    """
    # Read the list of image paths from the file
    with open(image_list_path, 'r') as file:
        image_paths = file.readlines()

    # Initialize lists to store all RGB values of all images
    all_images_rgb_values = []
    image_stats = []  # List to store stats for each image

    missing_images = []
    for image_path in tqdm(image_paths, desc="Processing Images"):
        image_path = image_path.strip()
        image_name = os.path.basename(image_path)
        expected_path = os.path.join(subfolder_path, image_name)

        if not os.path.exists(expected_path):
            missing_images.append(image_path)
        else:
            # Extract RGB values and calculate stats
            rgb_values = get_image_rgb_values(expected_path)
            mean = np.mean(rgb_values, axis=0)
            std = np.std(rgb_values, axis=0)
            all_images_rgb_values.append(rgb_values)
            # Append stats for the current image to the list
            image_stats.append([image_name] + mean.tolist() + std.tolist())

    # Write image statistics to the output CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Mean R', 'Mean G', 'Mean B', 'STD R', 'STD G', 'STD B'])
        csvwriter.writerows(image_stats)

    print(f"Results have been saved to {output_csv_path}")

    # Report missing images
    if missing_images:
        print("The following images are listed in 'imagelist.txt' but were not found in the specified subfolder:")
        for missing in missing_images:
            print(missing)

def get_image_rgb_values(image_path):
    """
    Extracts and returns the RGB values of an image as a numpy array.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] > 3:  # Image with alpha channel
        image_array = image_array[:, :, :3]
    
    return image_array.reshape((-1, 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate existence of images, calculate RGB statistics, and save to CSV.")
    parser.add_argument("--image_list_path", type=str, help="Path to the file containing the image list.")
    parser.add_argument("--subfolder_path", type=str, help="Path to the subfolder to check for image existence.")
    parser.add_argument("--output_csv_path", type=str, help="Path to the output CSV file for saving the results.")

    args = parser.parse_args()

    validate_images_and_calculate_stats(args.image_list_path, args.subfolder_path, args.output_csv_path)


# 
# python read_imagelist_RGB_MeanStdAll.py --image_list_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\imagelist.txt --subfolder_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images 
