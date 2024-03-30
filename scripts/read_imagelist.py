import os
import argparse
import utils



def validate_images(image_list_path, subfolder_path):
    """
    Validates if images listed in image_list_path exist in subfolder_path.

    Args:
    image_list_path (str): Path to the file containing the image list.
    subfolder_path (str): Path to the subfolder to check for image existence.
    """
    # Read the list of image paths from the file
    with open(image_list_path, 'r') as file:
        image_paths = file.readlines()

    # Normalize paths and check if each image exists in the subfolder
    missing_images = []
    for image_path in image_paths:
        image_path = image_path.strip()  # Remove any leading/trailing whitespace
        # Construct the expected path within the subfolder
        image_name = os.path.basename(image_path)
        expected_path = os.path.join(subfolder_path, image_name)

        if not os.path.exists(expected_path):
            missing_images.append(image_path)

    # Report the results
    if missing_images:
        print("The following images are listed in 'imagelist.txt' but were not found in the specified subfolder:")
        for missing in missing_images:
            print(missing)
    else:
        print("All images listed in 'imagelist.txt' were found in the specified subfolder.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate existence of images listed in a text file within a specified subfolder.")
    parser.add_argument("--image_list_path", type=str, help="Path to the file containing the image list.")
    parser.add_argument("--subfolder_path", type=str, help="Path to the subfolder to check for image existence.")

    args = parser.parse_args()

    # Perform the validation
    validate_images(args.image_list_path, args.subfolder_path)

  # example
  # python read_imagelist.py --image_list_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\imagelist.txt --subfolder_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images