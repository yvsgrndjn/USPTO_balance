import os
import sys
import time
import logging
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_build_dataset import (BASE_DIR, FOLDER_PATH, DATASET_NAME, DF_TEMP_PATH, PATH_TO_SAVE_FULL_CSV, PATH_TO_REF_DATA, TARGET_RXNS_PER_TEMPLATE, PATH_TO_SAVE_SUBSET_SPLIT, TRAIN_SET_PROP)

from src.uspto_balance.build_dataset import BuildDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build dataset and prepare a subset with train/valid/test splits from saved reactions stored during the enrichment process')

    # Optional arguments to override config settings
    parser.add_argument('--folder-path', type=str, default=FOLDER_PATH, help='Folder from which the equilibration was run')
    parser.add_argument('--dataset-name', type=str, default=DATASET_NAME, help='Name of the dataset')
    parser.add_argument('--df-temp-path', type=str, default=DF_TEMP_PATH, help='Path to the dataframe containing the templates')
    parser.add_argument('--path-to-save-full-csv', type=str, default=PATH_TO_SAVE_FULL_CSV, help='Path to save the csv dataset of all created and validated reactions')
    parser.add_argument('--path-to-ref-data', type=str, default=PATH_TO_REF_DATA, help='Path to the reference reactions dataset to check if a reaction is already in the dataset/avoid data leakage')
    parser.add_argument('--target-rxns-per-template', type=int, default=TARGET_RXNS_PER_TEMPLATE, help='Target number of reactions per template in the final dataset')
    parser.add_argument('--path-to-save-subset-split', type=str, default=PATH_TO_SAVE_SUBSET_SPLIT, help='Path to save the csv dataset of all created and validated reactions with a target number of reactions per template')
    parser.add_argument('--train-size', type=float, default=TRAIN_SET_PROP, help='Proportion of the dataset to include in the training set')
    parser.add_argument('--skip-appending-rxns-part', type=bool, default=False, help='Skip the appending reactions part and only split the dataset')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_arguments()

    # initialize the BuildDataset class
    build_dataset = BuildDataset(args.folder_path, 
                                 args.dataset_name, 
                                 args.df_temp_path, 
                                 args.path_to_save_full_csv, 
                                 args.path_to_ref_data, 
                                 args.target_rxns_per_template, 
                                 args.path_to_save_subset_split, 
                                 args.train_size, 
                                 args.skip_appending_rxns_part)
    
    if args.skip_appending_rxns_part:
        build_dataset.split_dataset()
    else:
        build_dataset.get_dataset()
        build_dataset.split_dataset()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f'Execution time: {end_time - start_time} seconds')