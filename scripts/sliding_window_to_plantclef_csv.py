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

def process_csv_and_analyze_classes_with_probabilities( args):
    # Initialize a defaultdict to group rows by filename
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    print("classes", len(cid_to_spid))
    spid_to_sp = load_species_mapping(args.species_mapping)
    
    data_by_filename = defaultdict(list)
    
    input_file_path = args.result
    input_file_path2 = args.result2
    thres98 = int(args.thres)
    
    file_out = open('D:\\PlantCLEF2024\\annotated\\dinoV2_results_thres.csv', 'w')
    
    
    
    file_out.write("plot_id;species_ids"+str('\n') )
    
    # Open the CSV file
    with open(input_file_path, mode='r') as file:
        # Create a DictReader object to read the CSV file
        reader = csv.DictReader(file, delimiter=',')
        
        # Group rows by filename
        for row in reader:
            data_by_filename[row['filename']].append(row)
            print(row)
            raise ValueError('A very specific bad thing happened.')
            
    # Open the CSV file 2
 
    with open(input_file_path2, mode='r') as file2:
        # Create a DictReader object to read the CSV file
        reader2 = csv.DictReader(file2, delimiter=',')
        
        # Group rows by filename
        for row in reader2:
            data_by_filename[row['filename']].append(row)
            
        
        
             
    # Analysis for each filename
    for filename, rows in data_by_filename.items():
        # Use dictionaries to track unique class indices with their probabilities
        # This allows separate tracking for class_index_1 and class_index_2
        class_probabilities = defaultdict(list)
    
        # Populate dictionaries with class indices and their probabilities
        min_threshold = 30
        max_classes = 8
        
        
        for row in rows:
            prob = float(row['probability_1'])
           # print("prob", prob)
            if prob > min_threshold:  # Consider only probabilities above the minimum threshold
                class_probabilities[row['class_index_1']].append(prob)
    
        species_id_set = set()
    
        # Sort class indices based on their highest probability and consider only top classes
        sorted_class_indices = sorted(class_probabilities, key=lambda x: max(class_probabilities[x]), reverse=True)[:max_classes]
    
        # Perform analysis for each unique class index within the top classes
        for class_index in sorted_class_indices:
            probabilities = class_probabilities[class_index]
            percentile_90 = np.percentile(probabilities, 98)
    
            # Add class indices with probabilities at or above the 90th percentile
            if any(prob >= percentile_90 for prob in probabilities):
                species_id_set.add(str(len(probabilities)))
                #species_id_set.add(int(cid_to_spid[int(class_index)]))  # Map class index to species ID and add to set
                #species_id_set.add(int(class_index))  #New version Map class index to species ID and add to set

        str_result = f"{filename.rstrip()};{list(species_id_set)}\n"

        file_out.write(str_result)

    file_out.close()
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


def main(args, DEBUG=True):
    
    #cid_to_spid = load_class_mapping(args.class_mapping)
   # print("classes", len(cid_to_spid))
    #spid_to_sp = load_species_mapping(args.species_mapping)
    # process_csv(args.result)
    # process_filename_and_analyze(args.result)
    process_csv_and_analyze_classes_with_probabilities(args)
    
    #cid_to_spid = load_class_mapping(args.class_mapping)
    
   # print("classes", len(cid_to_spid))
    
    #spid_to_sp = load_species_mapping(args.species_mapping)
    
    
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
    input_file_path = 'D:\\pretrained_models\\ResultsAll_518W_172S_150B.csv'  
    input_file_path2 = 'D:\\pretrained_models\\ResultsAll_172W_172S_150B.csv'
    
    #input_file_path = 'D:\\PlantCLEF2024\\annotated\\b518_S172_ALL.txt'
    #input_file_path = 'D:\\PlantCLEF2024\\annotated\\species_identificationsP1.txt'
    #input_file_path2 = ' '
    
    
    parser.add_argument("--result", type=str, default=input_file_path  )
    parser.add_argument("--result2", type=str, default=input_file_path2  )
    
    parser.add_argument("--thres", type=str, default="50"  )
    
    args = parser.parse_args()
    
    # Debug the trained model weights and arch , are they loading correctly yes
    # modeWeightCheck(args)
    
    #

    main(args, DEBUG=False)



