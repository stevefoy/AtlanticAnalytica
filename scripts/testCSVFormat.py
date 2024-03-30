# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:15:22 2024

@author: stevf
"""
import pandas as pd
import csv

# Function to read and check the data
def process_csv(input_file_path, output_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file_path, sep=';', quoting=csv.QUOTE_NONE)
    
    # Placeholder for checking or manipulating the DataFrame
    # Example: df['new_column'] = df['existing_column'].apply(lambda x: x*2)
    # This is where you can add your data checking or manipulation logic
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file_path, sep=';', index=False, quoting=csv.QUOTE_NONE)

# Example usage
input_file_path = 'D:\\pretrained_models\\result_crop800_slide259.csv'  # Replace with your actual input file path
output_file_path = 'D:\\pretrained_models\\checked_and_modified.csv'  # Replace with your desired output file path
process_csv(input_file_path, output_file_path)



