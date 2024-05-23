# -*- coding: utf-8 -*-
"""
Final version 

@author: Stehen Foy
"""
from argparse import ArgumentParser
import pandas as pd
import csv
import numpy as np
import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def process_csv(input_file_path):
    with open(input_file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',', fieldnames=[
            "filename", "x1", "y1", "x2", "y2", "crop_index",
            "class_index_1", "probability_1",
            "class_index_2", "probability_2",
            "class_index_3", "probability_3",
            "class_index_4", "probability_4",
            "class_index_5", "probability_5"
        ])
        for row in reader:
            print(row)  # Print each row to see its structure

def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name

def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return df['species'].to_dict()

def read_filenames(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def group_filenames(filenames):
    file_groups = defaultdict(list)
    for filename in filenames:
        parts = filename.split('-')
        identifier = parts[0] + '-' + parts[1] + '-' + parts[2]
        file_groups[identifier].append(filename)
    return file_groups

def preprocess_image_for_rocks(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def detect_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(kp1, desc1, kp2, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    angles = []
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    mean_angle = np.mean(angles)
    return mean_angle

def find_rotation(reference_image, images, tryRock=True):
    base = preprocess_image_for_rocks(reference_image) if tryRock else cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = detect_features(base)
    rotations = []
    for img in images:
        comparison = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        kp, desc = detect_features(comparison)
        angle = match_features(keypoints, descriptors, kp, desc)
        rotations.append(angle)
    return rotations

def correct_rotation(images, rotation_angles):
    corrected_images = []
    for img, angle in zip(images, rotation_angles):
        corrected_img = img.rotate(-angle, expand=True)
        corrected_images.append(corrected_img)
    return corrected_images

def create_mosaic(image_files, tile_size=(250, 250), matcher=False):
    images, texts = [], []
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                resized_img = img.resize(tile_size)
                date_text = image_file.split('-')[3]
                images.append(resized_img)
                texts.append(f"{date_text[:4]} {date_text[4:6]} {date_text[6:8]}")
        except IOError:
            print(f"Cannot open image {image_file}")
            continue

    if matcher and len(images) > 1:
        rotations = find_rotation(images[0], images[1:])
        images = correct_rotation(images, [0] + rotations)

    num_images = len(images)
    if num_images == 0:
        return None
    
    grid_size = int(num_images**0.5)
    if grid_size**2 < num_images:
        grid_size += 1

    mosaic = Image.new('RGB', (grid_size * tile_size[0], grid_size * tile_size[1]))
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for i, (image, text) in enumerate(zip(images, texts)):
        x = (i % grid_size) * tile_size[0]
        y = (i // grid_size) * tile_size[1]
        mosaic.paste(image, (x, y))
        draw = ImageDraw.Draw(mosaic)
        text_position = (x + 5, y + tile_size[1] - 20)
        draw.text(text_position, text, font=font, fill="white")

    return mosaic

def process_images(grouped_filenames, image_dir):
    for identifier, files in grouped_filenames.items():
        print(f"Processing group: {identifier} with {len(files)} files")
        image_paths = [os.path.join(image_dir, file) for file in files]
        mosaic = create_mosaic(image_paths)
        if mosaic:
            mosaic.show()
        else:
            print(f"No images could be processed for group {identifier}")

def setupProcessing(data_by_filename, args):
    base_directory_sam = r'D:\pretrained_models\segment-anything-main\SAM_Results'
    output_hugginface = r'D:\PlantCLEF2024\annotated\dinoV2_results_thresv2.csv'
    file_out = open(output_hugginface, 'w')
    highest_probabilities = defaultdict(lambda: defaultdict(float))  
    file_out.write("plot_id;species_ids\n")
    
    min_threshold = int(args.thres)

    for filename, rows in tqdm(data_by_filename.items()):
        class_probabilities = defaultdict(list)
        image_masks_folder = os.path.join(base_directory_sam, filename)
        mask_path = os.path.join(image_masks_folder, "maskRocks.png")
        file_mask_exists = os.path.exists(mask_path)
        
        if file_mask_exists:
            img = Image.open(mask_path).convert('L')
            threshold = 128
            binary_mask = np.where(np.array(img) > threshold, 255, 0)
            binary_mask_1or0 = np.where(np.array(img) > threshold, 1, 0)
            total_img_pixels = binary_mask_1or0.size
            white_pixels = np.sum(binary_mask_1or0)
            percentage_white = int((white_pixels / total_img_pixels) * 100)
            
            if percentage_white > 80:
                file_mask_exists = False
                print("bad file:", mask_path)

        if filename not in highest_probabilities:
            highest_probabilities[filename] = defaultdict(float)    

        for row in rows:
            for i in range(1, 2):
                prob = float(row[f'probability_{i}'])
                class_index = row[f'class_index_{i}']
                
                percentage_white = 0
                if file_mask_exists:
                    x1, y1, x2, y2 = (int(row[f'x1']), int(row[f'y1']), int(row[f'x2']), int(row[f'y2']))
                    roi = binary_mask[y1:y2, x1:x2]
                    total_pixels = roi.size
                    white_pixels = np.sum(roi == 255)
                    percentage_white = int((white_pixels / total_pixels) * 100)

                if prob > min_threshold and percentage_white < 60:
                    current_max_prob = class_probabilities.get(class_index, 0)
                    if prob > current_max_prob:
                        class_probabilities[class_index] = prob   

        max_classes = 8
        species_id_set = set()
        sorted_class_indices = sorted(class_probabilities, key=lambda x: class_probabilities[x], reverse=True)[:max_classes]
        threshold = 0.98
        
        for class_index in sorted_class_indices:
            probability = class_probabilities[class_index]
            if probability >= threshold:
                species_id_set.add(int(class_index))

        str_result = f"{filename.rstrip()};{list(species_id_set)}\n"
        file_out.write(str_result)

    file_out.close()

def setupBBFiles(args):
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes loaded:", len(cid_to_spid))
    data_by_filename = defaultdict(list)
    filetypes_class_index = [
        r'D:\pretrained_models\ResultsAll_518W_172S_150B.csv',  
        r'D:\pretrained_models\ResultsAll_172W_172S_150B.csv'   
    ]
    new_filetypes_class_index = [r'D:\pretrained_models\ResultsAll_224W_112S_50B.csv']

   

    for fileName in filetypes_class_index:
           with open(fileName, mode='r') as file:
               reader = csv.DictReader(file, delimiter=',')
               for row in reader:
                   for i in range(1, 6):
                       class_index = row[f'class_index_{i}']
                       row[f'class_index_{i}'] = int(cid_to_spid[int(class_index)])
                   data_by_filename[row['filename']].append(row)
    
    for fileName in new_filetypes_class_index:
        with open(fileName, mode='r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                for i in range(1, 6):
                    spid_index = row[f'class_index_{i}']
                    row[f'class_index_{i}'] = int(spid_index)
                data_by_filename[row['filename']].append(row)

    return data_by_filename

def plot_class_probabilities(class_probabilities_all):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for class_index, data in class_probabilities_all.items():
        ax[0].scatter(data['max_prob'], data['count'], label=f'Class {class_index}')

    ax[0].set_xlabel('Max Probability')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Max Probability vs Count per Class')
    ax[0].legend()

    # Plot histograms for max_prob and count
    all_max_probs = [prob for data in class_probabilities_all.values() for prob in data['max_prob']]
    all_counts = [count for data in class_probabilities_all.values() for count in data['count']]

    ax[1].hist(all_max_probs, bins=20, alpha=0.5, label='Max Probability')
    ax[1].hist(all_counts, bins=20, alpha=0.5, label='Count')
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Histograms of Max Probability and Count')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def adjust_max_classes(probabilities, avg_count, current_count):
    # Adjust the number of classes based on the current count relative to the average count
    if current_count > avg_count:
        return 12 #
    elif current_count > avg_count / 2:
        return 9
    elif current_count > avg_count / 3:
        return 7
    else:
        return 6
    
def groupProcess(data_by_filename, args):
    file_path = r"D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\imagelist.txt"
    image_full_path = r"D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images"
    base_directory_sam = r"D:\pretrained_models\segment-anything-main\SAM_Results"
    
    
    filenames = read_filenames(file_path)
    grouped_filenames = group_filenames(filenames)
    min_threshold = 48
    max_classes = 9
    output_hugginface = r'D:\PlantCLEF2024\annotated\dinoV2_results_thresv18.csv'
    file_out = open(output_hugginface, 'w')
    file_out.write("plot_id;species_ids\n")
    all_probs = []

    for identifier, files in grouped_filenames.items():
        print(f"Processing group: {identifier} with {len(files)} files")
        filename_images_group = [file.replace(".jpg", "") for file in files]
        class_probabilities = defaultdict(lambda: {'max_prob': 0, 'count': 0})
        
        group_count_all = []
        for image_filename in filename_images_group:
            imagefile_data = data_by_filename[image_filename]
            image_masks_folder = os.path.join(base_directory_sam, image_filename)
            mask_path = os.path.join(image_masks_folder, "maskRocks.png")
            file_mask_exists = os.path.exists(mask_path)

            if file_mask_exists:
                img = Image.open(mask_path).convert('L')
                threshold = 128
                binary_mask = np.where(np.array(img) > threshold, 255, 0)
                binary_mask_1or0 = np.where(np.array(img) > threshold, 1, 0)
                total_img_pixels = binary_mask_1or0.size
                white_pixels = np.sum(binary_mask_1or0)
                percentage_white = int((white_pixels / total_img_pixels) * 100)
                if percentage_white > 80:
                    file_mask_exists = False
                    print("bad file:", mask_path)
            
            all_counts = 0
            for row in imagefile_data:
                for i in range(1, 6):
                    prob = float(row[f'probability_{i}'])
                    class_index = row[f'class_index_{i}']
                    x1, y1, x2, y2 = (int(row[f'x1']), int(row[f'y1']), int(row[f'x2']), int(row[f'y2']))
                   # print(x1," ", y1," ", x2," ", y2)
                    percentage_white = 0
                    
                    if file_mask_exists:
                        roi = binary_mask[y1:y2, x1:x2]
                        total_pixels = roi.size
                        white_pixels = np.sum(roi == 255)
                        percentage_white = int((white_pixels / total_pixels) * 100)

                    #if prob > min_threshold and percentage_white < 60:
                    if prob > min_threshold and percentage_white < 50:
                        all_probs.append(prob)
                        
                        if prob > class_probabilities[class_index]['max_prob']:
                            class_probabilities[class_index]['max_prob'] = prob
                        class_probabilities[class_index]['count'] += 1
                        all_counts +=1
        
            group_count_all.append(all_counts)
            
            
                 
        avg_count_group = np.mean(all_counts)
        index_current_cout = 0
        
        for image_filename, current_cout in zip(filename_images_group, group_count_all) :
            
            max_classes = adjust_max_classes(all_probs, avg_count_group, current_cout)
    
            sorted_class_indices = sorted(
                class_probabilities.keys(),
                key=lambda x: (class_probabilities[x]['max_prob'] * (1 + np.log1p(class_probabilities[x]['count']))),
                reverse=True
            )[:max_classes]
    
            print(f"Top {max_classes} classes for group {identifier}: {sorted_class_indices}")
            str_result = f"{image_filename};{list(sorted_class_indices)}\n"
            file_out.write(str_result)
            index_current_cout +=1

    file_out.close()
    
    return class_probabilities

def main(args):
    data_by_filename = setupBBFiles(args)
    results = groupProcess(data_by_filename, args)
    plot_class_probabilities(results)
    
    # Analysis on image 
    filenames = read_filenames(r"D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\imagelist.txt")
    grouped_filenames = group_filenames(filenames)
    print("Finshed ")
   # process_images(grouped_filenames, "D:\PlantCLEF2024\PlantCLEF2024\PlantCLEF2024test\images")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--class_mapping", type=str, default=r'D:\PlantCLEF2024\PlantCLEF2024\class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default=r'D:\PlantCLEF2024\PlantCLEF2024\species_id_to_name.txt')
    parser.add_argument("--result", type=str, default=r'D:\PlantCLEF2024\annotated\b518_S172_ALL.txt')
    parser.add_argument("--result2", type=str, default=' ')
    parser.add_argument("--thres", type=str, default="50")
    
    args = parser.parse_args()
    main(args)
