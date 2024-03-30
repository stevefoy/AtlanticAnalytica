# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:15:22 2024

@author: stevf
"""
from argparse import ArgumentParser
import pandas as pd
import csv
import numpy as np

import csv


import csv
from collections import defaultdict
import csv
from collections import defaultdict
import numpy as np

def process_csv_and_analyze_classes_with_probabilities(input_file_path):
    # Initialize a defaultdict to group rows by filename
    data_by_filename = defaultdict(list)
    
    # Open the CSV file
    with open(input_file_path, mode='r') as file:
        # Create a DictReader object to read the CSV file
        reader = csv.DictReader(file, delimiter=',')
        
        # Group rows by filename
        for row in reader:
            data_by_filename[row['filename']].append(row)
    
    # Analysis for each filename
    for filename, rows in data_by_filename.items():
        # Use dictionaries to track unique class indices with their probabilities
        # This allows separate tracking for class_index_1 and class_index_2
        class_probabilities = defaultdict(list)
        
        # Populate the dictionaries with class indices and their probabilities
        for row in rows:
            if row['class_index_1']:
                class_probabilities[(row['class_index_1'], 'probability_1')].append(float(row['probability_1']))
            if row['class_index_2']:
                class_probabilities[(row['class_index_2'], 'probability_2')].append(float(row['probability_2']))
        
        # Perform analysis for each unique class index
        for (class_index, prob_key), probabilities in class_probabilities.items():
            # Calculate the 90th percentile for the probabilities
            percentile_90 = np.percentile(probabilities, 90)
            
            # Output the result
            print(f"{filename} - Class Index: {class_index} with {prob_key}: 90th percentile = {percentile_90:.2f}")
            
            # Optionally, for rows in the current class index group, print details if their probability is above the 90th percentile
            print("Entries above the 90th percentile:")
            for row in rows:
                if ((row['class_index_1'] == class_index and prob_key == 'probability_1') or 
                    (row['class_index_2'] == class_index and prob_key == 'probability_2')) and float(row[prob_key]) >= percentile_90:
                        print(f"  {class_index} with {prob_key}: {row[prob_key]}")




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


def main(args, DEBUG=True):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    # process_csv(args.result)
    # process_filename_and_analyze(args.result)
    process_csv_and_analyze_classes_with_probabilities(args.result)
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    
    print("classes", len(cid_to_spid))
    
    spid_to_sp = load_species_mapping(args.species_mapping)
    
    
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

    input_file_path = 'D:\\pretrained_models\\tempResults.csv'  
    parser.add_argument("--result", type=str, default=input_file_path  )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)



