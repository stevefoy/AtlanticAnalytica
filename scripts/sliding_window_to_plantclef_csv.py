# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Final version 

@author: stevf
"""
from argparse import ArgumentParser
import pandas as pd
import csv
import numpy as np

import csv
import os

import csv
from collections import defaultdict
import csv
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

"""
def process_csv_and_analyze_classes_with_probabilities( args):
    # Initialize a defaultdict to group rows by filename
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    
    data_by_filename = defaultdict(list)
    
    # SAM 
 
    # csv_file = "D:\\pretrained_models\\bb224_s112_R3_Test.csv"
    csv_file3 = r"D:\pretrained_models\bb224_s112_R3_Test.csv"
    csv_file1 = r'D:\pretrained_models\ResultsAll_518W_172S_150B.csv'  
    csv_file2 = r'D:\pretrained_models\ResultsAll_172W_172S_150B.csv'
    
    filetypes_class_index = [csv_file1, csv_file2 ]
    new_filetypes_class_index = [csv_file3]
    #new_filetypes_class_index = []
    
    thres98 = int(args.thres)
    
    
    
    
    # SAVE DATA FOR HUGGINFACE PUSH
    output_hugginface =  'D:\\PlantCLEF2024\\annotated\\dinoV2_results_thresv2.csv'
    file_out = open(output_hugginface, 'w')
    
    
    
    file_out.write("plot_id;species_ids"+str('\n') )
    

         
    # Open the CSV file 2
    for fileName in filetypes_class_index:
        with open(fileName , mode='r') as file:
            # Create a DictReader object to read the CSV file
            reader = csv.DictReader(file, delimiter=',')
            # Group rows by filename
            for row in reader:
                for i in range(1, 6):  # Assuming up to 5 classes per crop
                    #prob = float(row[f'probability_{i}'])
                    class_index = row[f'class_index_{i}']
                    row[f'class_index_{i}']=int(cid_to_spid[int(class_index)])
                    
                
                data_by_filename[row['filename']].append(row)
                #print(row)
                #raise ValueError('A very specific bad thing happened.')
    
    # Open the CSV file
    for fileName in new_filetypes_class_index:
        with open(fileName , mode='r') as file:
            # Create a DictReader object to read the CSV file
            reader = csv.DictReader(file, delimiter=',')
            # Group rows by filename
            for row in reader:
                
                data_by_filename[row['filename']].append(row)
                #print(row)
                #raise ValueError('A very specific bad thing happened.')
    
     
             
"""
'''
def process_filename_and_analyze(input_file_path):
    # Initialize a defaultdict to group rows by filename
    data_by_filename = defaultdict(list)
    
    # Open the CSV file
    with open(input_file_path, mode='r') as file:
        # Create a DictReader object to read the CSV file
        reader = csv.DictReader(file, delimiter=',')
    
        # Group rows by filename
        for row in reader:
            data_by_filename[row['filename']].append(row)
    
        # Analysis: Find the 90th percentile of probability_1 for each filename group
        for filename, rows in data_by_filename.items():
            # Extract probability_1 values and convert them to floats
            probabilities = [float(row['probability_1']) for row in rows if row['probability_1']]
            
            # Calculate the 90th percentile
            percentile_90 = np.percentile(probabilities, 90)
            
            # Output the result
            print(f"{filename}: 90th percentile of Probability 1 = {percentile_90:.2f}")
            
            # For each row in the current filename group, print details if its probability_1 is above the 90th percentile
            print("Details of entries above the 90th percentile:")
            for row in rows:
                if float(row['probability_1']) >= percentile_90:
                    print(f"  Class Index 1: {row['class_index_1']}, Probability 1: {row['probability_1']}")
'''

def process_csv(input_file_path):
    # Open the CSV file
    with open(input_file_path, mode='r') as file:
        # Create a DictReader object to read the CSV file
        reader = csv.DictReader(file, delimiter=',', fieldnames=[
            "filename", "x1", "y1", "x2", "y2", "crop_index",
            "class_index_1", "probability_1",
            "class_index_2", "probability_2",
            "class_index_3", "probability_3",
            "class_index_4", "probability_4",
            "class_index_5", "probability_5"
        ])

        # Loop through each row in the CSV
        for row in reader:
            # `row` is a dictionary where the column names are keys
            # Example of accessing data: row['filename'] or row['x1']
            
            # Perform your data processing here. For demonstration:
            print(row)  # Print each row to see its structure

# Example usage:
# process_csv('path/to/your/file.csv')



def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

from PIL import Image
import os

def read_filenames(file_path):
    """Read filenames from a specified text file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def group_filenames(filenames):
    """Group filenames by the first and second parts of their identifiers."""
    file_groups = defaultdict(list)
    for filename in filenames:
        parts = filename.split('-')
        identifier = parts[0] + '-' + parts[1] + '-' + parts[2]
        file_groups[identifier].append(filename)
        
    return file_groups

from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

def preprocess_image_for_rocks(image):
    """Preprocess the image to enhance grey rocks by using edge detection."""
    # Convert PIL image to numpy array and then to grayscale
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Use Canny edge detection to emphasize edges (tune these thresholds as needed)
    if True:
        edges = cv2.Canny(gray, 100, 200)
    else:
        # Visualize the grayscale image
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.title('Grayscale Image')
        plt.imshow(gray, cmap='gray')
        
        # Use Canny edge detection to emphasize edges (tune these thresholds as needed)
        edges = cv2.Canny(gray, 100, 200)
        
        # Visualize the edges
        plt.subplot(1, 2, 2)
        plt.title('Edge Detection')
        plt.imshow(edges, cmap='gray')
        plt.show()

    return edges


def find_rotation(reference_image, images, tryRock = True):
    # Convert PIL images to grayscale arrays for processing
    
    if tryRock:
        base = preprocess_image_for_rocks(reference_image)
    else:
        base = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2GRAY)
        
    
    keypoints, descriptors = detect_features(base)
    
    rotations = []
    for img in images:
        comparison = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        kp, desc = detect_features(comparison)
        angle = match_features(keypoints, descriptors, kp, desc)
        rotations.append(angle)
    
    return rotations

def detect_features(image):
    # Using ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(kp1, desc1, kp2, desc2):
    # Create matcher and find matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    angles = []
    
    for match in matches:
        # Indexes of the keypoints in the matches
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        # x - coordinates of keypoints in both images
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt
        
        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Average angle of all matches
    mean_angle = np.mean(angles)
    return mean_angle



def correct_rotation(images, rotation_angles):
    corrected_images = []
    for img, angle in zip(images, rotation_angles):
        print(f"Correcting rotation for image by angle: {angle}")  # Debug: Check rotation angles
        corrected_img = img.rotate(-angle, expand=True)  # Correcting the rotation
        corrected_images.append(corrected_img)
       # corrected_img.show()  # Debug: Show each corrected image
    return corrected_images

def create_mosaic_matcher(image_files, tile_size=(250, 250)):
    """Create a mosaic image from a list of image paths, adding date text from filename."""
    images = []
    texts = []
    
    # Open images and resize them
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                resized_img = img.resize(tile_size)
                
                
                # Extract the date from the filename
                date_text = image_file.split('-')[3]  # Assuming the date is the fourth part
                images.append(resized_img)
                texts.append(date_text[:4]+str(" ")+date_text[4:6]+str(" ")+date_text[6:8])  # Assuming the format is YYYYMMDD without extension
        except IOError:
            print(f"Cannot open image {image_file}")
            continue
    
    

    rotations = find_rotation(images[0], images[1:])  # First image is the base
    images = correct_rotation(images, [0] + rotations)
    
    # Number of images
    num_images = len(images)
    if num_images == 0:
        return None
    
    # Determine the grid size
    grid_size = int(num_images**0.5)
    if grid_size**2 < num_images:
        grid_size += 1
    
    # Create a new image large enough to contain all the tiles
    mosaic = Image.new('RGB', (grid_size * tile_size[0], grid_size * tile_size[1]))
    
    # Define a font for the text
    try:
        # For better compatibility, specify an absolute path to a TTF font file
        font = ImageFont.truetype("arial.ttf", 12)  # Adjust the font size as needed
    except IOError:
        print("Font file not found, using default font.")
        font = ImageFont.load_default()
    
    # Paste images into the mosaic and add text
    for i, (image, text) in enumerate(zip(images, texts)):
        x = (i % grid_size) * tile_size[0]
        y = (i // grid_size) * tile_size[1]
        # image.show()  # Debug: Show each corrected image
        mosaic.paste(image, (x, y))
        # Create a draw object
        draw = ImageDraw.Draw(mosaic)
        # Position the text at the bottom center of each image
        text_position = (x + 5, y + tile_size[1] - 20)  # Adjust positioning as needed
        draw.text(text_position, text, font=font, fill="white")  # Change text color if needed
    
    return mosaic

def create_mosaic_simple(image_files, tile_size=(250, 250)):
    """Create a mosaic image from a list of image paths, adding date text from filename."""
    images = []
    texts = []
    
    # Open images and resize them
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                resized_img = img.resize(tile_size)
                # Extract the date from the filename
                date_text = image_file.split('-')[3]  # Assuming the date is the fourth part
                images.append(resized_img)
                texts.append(date_text[:4]+str(" ")+date_text[4:6]+str(" ")+date_text[6:8])  # Assuming the format is YYYYMMDD without extension
        except IOError:
            print(f"Cannot open image {image_file}")
            continue
    
    # Number of images
    num_images = len(images)
    if num_images == 0:
        return None
    
    # Determine the grid size
    grid_size = int(num_images**0.5)
    if grid_size**2 < num_images:
        grid_size += 1
    
    # Create a new image large enough to contain all the tiles
    mosaic = Image.new('RGB', (grid_size * tile_size[0], grid_size * tile_size[1]))
    
    # Define a font for the text
    try:
        # For better compatibility, specify an absolute path to a TTF font file
        font = ImageFont.truetype("arial.ttf", 12)  # Adjust the font size as needed
    except IOError:
        print("Font file not found, using default font.")
        font = ImageFont.load_default()
    
    # Paste images into the mosaic and add text
    for i, (image, text) in enumerate(zip(images, texts)):
        x = (i % grid_size) * tile_size[0]
        y = (i // grid_size) * tile_size[1]
        mosaic.paste(image, (x, y))
        # Create a draw object
        draw = ImageDraw.Draw(mosaic)
        # Position the text at the bottom center of each image
        text_position = (x + 5, y + tile_size[1] - 20)  # Adjust positioning as needed
        draw.text(text_position, text, font=font, fill="white")  # Change text color if needed
    
    return mosaic


def process_images(grouped_filenames, image_dir):
    """Process images for each group using PIL, creating mosaics."""
    for identifier, files in grouped_filenames.items():
        print(f"Processing group: {identifier} with {len(files)} files")
        image_paths = [os.path.join(image_dir, file) for file in files]
        mosaic = create_mosaic_simple(image_paths)
        if mosaic:
            mosaic.show()  # Display the mosaic
        else:
            print(f"No images could be processed for group {identifier}")

def groupTest():


    # load the image files
    file_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\imagelist.txt"
    image_full_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\"
    image_files = []
    
    with open(file_path, 'r') as file:
    # Read all lines in the file and strip newline characters
        image_files = [image_full_path+line.strip() for line in file]
        


    filenames = read_filenames(file_path)
    grouped_filenames = group_filenames(filenames)
    process_images(grouped_filenames, image_full_path)

def setupProcessing(data_by_filename):
    # SAVE DATA FOR HUGGINFACE PUSH
    base_directory_sam = r"D:\pretrained_models\segment-anything-main\SAM_Results"

   
    output_hugginface =  'D:\\PlantCLEF2024\\annotated\\dinoV2_results_thresv2.csv'
    file_out = open(output_hugginface, 'w')
    highest_probabilities = defaultdict(lambda: defaultdict(float))  
    
    file_out.write("plot_id;species_ids"+str('\n') )
    
    # Analysis for each filename
    for filename, rows in tqdm(data_by_filename.items()):
        # Use dictionaries to track unique class indices with their probabilities
        # This allows separate tracking for class_index_1 and class_index_2
        class_probabilities = defaultdict(list)
        
        image_masks_folder = os.path.join(base_directory_sam, filename )
        mask_path = os.path.join(image_masks_folder, "maskRocks.png")
        
        file_mask_exists =  os.path.exists(mask_path)
        
        if file_mask_exists == True:
            #print("File here", mask_path )
            img = Image.open(mask_path).convert('L')
            threshold = 128
            binary_mask = np.where(np.array(img) > threshold, 255, 0)
            binary_mask_1or0 = np.where(np.array(img) > threshold, 1, 0)

            total_img_pixels = binary_mask_1or0.size
            white_pixels = np.sum(binary_mask_1or0)
            
            percentage_white =int( (white_pixels / total_img_pixels) * 100)
            
            
            if percentage_white > 80:
                # SKIP Files have have MAsk issues
                file_mask_exists = False
                print("bad file:", mask_path )
            

        else:
            pass
            #print("No file", mask_path )
    
        if filename not in highest_probabilities:
            highest_probabilities[filename] = defaultdict(float)    
        
        for row in rows:
            for i in range(1, 2):  # Assuming up to 5 classes per crop range(1, 6)
                prob = float(row[f'probability_{i}'])
                class_index = row[f'class_index_{i}']
                
                percentage_white = 0
                if file_mask_exists:
                    # Define the bounding box (x1, y1, x2, y2)
                    x1, y1, x2, y2 = (int(row[f'x1']), int(row[f'y1']), int(row[f'x2']), int(row[f'y2']))  # Replace x1, y1, x2, y2 with actual values
                    #print(x1, y1, x2, y2)
                    # Extract the region of interest from the binary mask
                    roi = binary_mask[y1:y2, x1:x2]
                    
                    # Calculate the percentage of pixels that are 255 in the ROI
                    total_pixels = roi.size
                    white_pixels = np.sum(roi == 255)
                    percentage_white =int( (white_pixels / total_pixels) * 100)
                    
                    #print("Percentage of white pixels:", percentage_white)

                if prob > min_threshold and percentage_white < 60:
                    current_max_prob = class_probabilities.get(class_index, 0)
                    if prob > current_max_prob:
                        class_probabilities[class_index] = prob   

        max_classes = 8  # Define or adjust as necessary
        species_id_set = set()

        sorted_class_indices = sorted(class_probabilities, key=lambda x: class_probabilities[x], reverse=True)[:max_classes]
        
        threshold = 0.98  # Define a threshold for selecting classes, adjust as needed
        
        for class_index in sorted_class_indices:
            probability = class_probabilities[class_index]
        
            # Add class index if the probability exceeds the threshold
            if probability >= threshold:
                species_id_set.add(int(class_index))
            

    
        str_result = f"{filename.rstrip()};{list(species_id_set)}\n"
        file_out.write(str_result)

    file_out.close()

def setupBBFiles():
    # Initialize a defaultdict to group rows by filename
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes loaded:", len(cid_to_spid))
    
    data_by_filename = defaultdict(list)
    


    
    # csv_file = "D:\\pretrained_models\\bb224_s112_R3_Test.csv"
    csv_file3 = r"D:\pretrained_models\bb224_s112_R3_Test.csv"
    csv_file4 = r"D:\pretrained_models\bb518_s112_R3_eql_HSV.csv"
    
    csv_file1 = r'D:\pretrained_models\ResultsAll_518W_172S_150B.csv'  
    csv_file2 = r'D:\pretrained_models\ResultsAll_172W_172S_150B.csv'
    
    filetypes_class_index = [csv_file1, csv_file2 ]
    new_filetypes_class_index = [ ]
    # new_filetypes_class_index = []


    # Open the CSV file 2
    for fileName in filetypes_class_index:
        with open(fileName , mode='r') as file:
            # Create a DictReader object to read the CSV file
            reader = csv.DictReader(file, delimiter=',')
            # Group rows by filename
            for row in reader:
                for i in range(1, 6):  # Assuming up to 5 classes per crop
                    #prob = float(row[f'probability_{i}'])
                    class_index = row[f'class_index_{i}']
                    row[f'class_index_{i}']=int(cid_to_spid[int(class_index)])
                    
                
                data_by_filename[row['filename']].append(row)
                #print(row)
                #raise ValueError('A very specific bad thing happened.')
    
    # Open the CSV file
    for fileName in new_filetypes_class_index:
        with open(fileName , mode='r') as file:
            # Create a DictReader object to read the CSV file
            reader = csv.DictReader(file, delimiter=',')
            # Group rows by filename
            for row in reader:
                for row in reader:
                    for i in range(1, 6):  # Assuming up to 5 classes per crop
                        #prob = float(row[f'probability_{i}'])
                        spid_index = row[f'class_index_{i}']
                        row[f'class_index_{i}']=int(spid_index)
                        
                    data_by_filename[row['filename']].append(row)
                
                data_by_filename[row['filename']].append(row)
                #print(row)
                #raise ValueError('A very specific bad thing happened.')
    
    
    
    return data_by_filename



def groupProcess(data_by_filename):


    # load the image files
    file_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\imagelist.txt"
    image_full_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\"
    base_directory_sam = r"D:\pretrained_models\segment-anything-main\SAM_Results"
    
    group = dict()

    filenames = read_filenames(file_path)
    grouped_filenames = group_filenames(filenames)
    
    min_threshold = 48
    max_classes = 9
    
    
    
    # SAVE DATA FOR HUGGINFACE PUSH
    output_hugginface =  r'D:\PlantCLEF2024\annotated\dinoV2_results_thresv20.csv'
    file_out = open(output_hugginface, 'w')
    file_out.write("plot_id;species_ids"+str('\n') )


    for identifier, files in grouped_filenames.items():
        print(f"Processing group: {identifier} with {len(files)} files")
        
        # Process a group of images of the same plot
        filename_images_group = [ file.replace(".jpg", "") for file in files]
        
        # AT plot level create a class prob list
        class_probabilities = defaultdict(lambda: {'max_prob': 0, 'count': 0})
        
        for image_filename in filename_images_group:
            imagefile_data = data_by_filename[image_filename]
            
            image_masks_folder = os.path.join(base_directory_sam, image_filename )
            mask_path = os.path.join(image_masks_folder, "maskRocks.png")
            
            file_mask_exists =  os.path.exists(mask_path)
            
            if file_mask_exists == True:
                #print("File here", mask_path )
                img = Image.open(mask_path).convert('L')
                threshold = 128
                binary_mask = np.where(np.array(img) > threshold, 255, 0)
                binary_mask_1or0 = np.where(np.array(img) > threshold, 1, 0)

                total_img_pixels = binary_mask_1or0.size
                white_pixels = np.sum(binary_mask_1or0)
                
                percentage_white =int( (white_pixels / total_img_pixels) * 100)
                
                
                if percentage_white > 80:
                    # SKIP Files have have MAsk issues
                    file_mask_exists = False
                    print("bad file:", mask_path )
                

            else:
                pass
                #print("No file", mask_path )

            # Print all data rows associated with the filename
            for row in imagefile_data:
                # print(row)
                # CAN do up to 5 classes per crop range(1, 6)
                for i in range(1, 6): 
                    prob = float(row[f'probability_{i}'])
                    class_index = row[f'class_index_{i}']
                    # Define the bounding box (x1, y1, x2, y2)
                    x1, y1, x2, y2 = (int(row[f'x1']), int(row[f'y1']), int(row[f'x2']), int(row[f'y2']))
                    
                    percentage_white = 0
                    if file_mask_exists:
                        # Define the bounding box (x1, y1, x2, y2)
                        x1, y1, x2, y2 = (int(row[f'x1']), int(row[f'y1']), int(row[f'x2']), int(row[f'y2']))  # Replace x1, y1, x2, y2 with actual values
                        #print(x1, y1, x2, y2)
                        # Extract the region of interest from the binary mask
                        roi = binary_mask[y1:y2, x1:x2]
                        
                        # Calculate the percentage of pixels that are 255 in the ROI
                        total_pixels = roi.size
                        white_pixels = np.sum(roi == 255)
                        percentage_white =int( (white_pixels / total_pixels) * 100)                    
                    
                    
                    
                    #print(x1, y1, x2, y2)
                
                    if prob > min_threshold and percentage_white < 60:
                        if prob > class_probabilities[class_index]['max_prob']:
                            class_probabilities[class_index]['max_prob'] = prob
                        
                        # Record a count anyway of the case
                        class_probabilities[class_index]['count'] += 1
                            
        # At group level sort highest classes
        #sorted_class_indices = sorted(class_probabilities, key=lambda x: class_probabilities[x], reverse=True)[:max_classes]
        sorted_class_indices = sorted(
            class_probabilities.keys(),
            key=lambda x: (class_probabilities[x]['max_prob'], class_probabilities[x]['count']),
            reverse=True
        )[:max_classes]
        
        print(f"Top {max_classes} classes {identifier}: {sorted_class_indices}")
        

        
        for image_filename in filename_images_group:
            
            print(f"Top {max_classes} classes for {image_filename}: {sorted_class_indices}")
            str_result = f"{image_filename};{list(sorted_class_indices)}\n"
            file_out.write(str_result)            
            
        #raise ValueError("STOP") 
        
    file_out.close()
    
    
    #process_images(grouped_filenames, image_full_path)

def main(args, DEBUG=True):
    
    # cid_to_spid = load_class_mapping(args.class_mapping)
    # print("classes", len(cid_to_spid))
    # spid_to_sp = load_species_mapping(args.species_mapping)
    # process_csv(args.result)
    # process_filename_and_analyze(args.result)
    
    
    
    
    
    # MAIN
    data_by_filename = setupBBFiles()
    groupProcess(data_by_filename)
    groupTest()
    # process_csv_and_analyze_classes_with_probabilities(args)
    
    
    # cid_to_spid = load_class_mapping(args.class_mapping) 
    # print("classes", len(cid_to_spid))   
    # spid_to_sp = load_species_mapping(args.species_mapping)
    
    
    # load the image files
    file_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\imagelist.txt"
    image_full_path = "D:\\PlantCLEF2024\\PlantCLEF2024\\PlantCLEF2024test\\images\\"
    image_files = []
    
    with open(file_path, 'r') as file:
    # Read all lines in the file and strip newline characters
        image_files = [image_full_path+line.strip() for line in file]
        
    
 


if __name__ == '__main__':
    parser = ArgumentParser()


    parser.add_argument("--class_mapping", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='D:\\PlantCLEF2024\\PlantCLEF2024\\species_id_to_name.txt')


    # Real test

    
    input_file_path = 'D:\\PlantCLEF2024\\annotated\\b518_S172_ALL.txt'
    input_file_path = 'D:\\PlantCLEF2024\\annotated\\species_identificationsP1.txt'
    input_file_path2 = ' '
    
    
    parser.add_argument("--result", type=str, default=input_file_path  )
    parser.add_argument("--result2", type=str, default=input_file_path2  )
    
    parser.add_argument("--thres", type=str, default="50"  )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)



