import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import utils

    
    
    
def validate_images_and_calculate_stats(image_list_path, subfolder_path):
    """
    Validates if images listed in image_list_path exist in subfolder_path, calculates their RGB mean and standard deviation,
    and aggregates this information to calculate the overall mean and STD for the folder.

    Args:
    image_list_path (str): Path to the file containing the image list.
    subfolder_path (str): Path to the subfolder to check for image existence.
    """
    

    all_images_rgb_paths = createImageList(image_list_path, subfolder_path)

    
    # Calculate and print the overall mean and STD for the folder
    print("Starting:ALL images, np.concat + np.mean and np.std")
    overall_mean, overall_std = incremental_mean_std(all_images_rgb_paths)  
    print(f"Overall Mean: {overall_mean}, Overall STD: {overall_std}")


def get_image_rgb_values(image_path):
    """
    Extracts and returns the RGB values of an image as a numpy array in float32 for memory efficiency.
    """
    # print(image_path)
    image = Image.open(image_path)
    
    image_array = np.array(image, dtype=np.float32)
    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] > 3:  # Image with alpha channel
        image_array = image_array[:, :, :3]
    
    return image_array.reshape((-1, 3))

def incremental_mean_std(image_paths):
    """
    Calculates the incremental mean and standard deviation for a list of image paths.
    """
    n = 0
    mean = np.zeros(3, dtype=np.float32)
    M2 = np.zeros(3, dtype=np.float32)
    
    for image_path in tqdm(image_paths, desc="Processing Images"):
        rgb_values = get_image_rgb_values(image_path).astype(np.float32)
        n += len(rgb_values)
        delta = rgb_values - mean
        mean += delta.sum(axis=0) / n
        delta2 = rgb_values - mean
        M2 += (delta * delta2).sum(axis=0)
    
    variance = M2 / n
    std = np.sqrt(variance)
    
    return mean, std


    


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate existence of images and calculate their aggregate RGB statistics.")
    parser.add_argument("--image_list_path", type=str, help="Path to the file containing the image list.")
    parser.add_argument("--subfolder_path", type=str, help="Path to the subfolder to check for image existence.")

    args = parser.parse_args()

    # Perform the validation and calculations
    validate_images_and_calculate_stats(args.image_list_path, args.subfolder_path)


# python read_imagelist_RGB_MeanStdAll.py --image_list_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\imagelist.txt --subfolder_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images
# Result of Test images Overall Mean: [114.83894 115.1152   88.3965 ], Overall STD: [59.05115764 57.78404997 58.27347664]
