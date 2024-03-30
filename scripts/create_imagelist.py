# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:15:09 2024

@author: stevf
"""

import os
import argparse

def create_image_list(folder_path, save_folder_path):
    """
    Iterates over all images in the specified folder and writes their paths to imagelist.txt.

    Args:
    folder_path (str): The path to the folder containing images.
    """
    
    # Define the image file extensions to look for
    image_extensions = ('.jpg', '.jpeg', '.png')
    file_name ='imagelist.txt'
    fullsave_folder_path = os.path.join(save_folder_path, file_name)
    
    # Open the output file in write mode
    with open(fullsave_folder_path, 'w') as output_file:
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(image_extensions):
                    # Write the full path of each image file to the output file
                    output_file.write( file + '\n')


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generate a list of image files in a folder.")
    
    # Add an argument for the folder path
    parser.add_argument("--folder_path", type=str, help="The path to the folder containing the images.")
    parser.add_argument("--save_path", type=str,default="./", help="The path to save listfile")
    
    # Parse the arguments
    args = parser.parse_args()
    
    
    
    
    # Call the function with the provided folder path
    create_image_list(args.folder_path, args.save_path)
    
    print("Image list has been created in imagelist.txt.")
    
    # Example 
    # create_imagelist.py --folder_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images  --save_path D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\
        