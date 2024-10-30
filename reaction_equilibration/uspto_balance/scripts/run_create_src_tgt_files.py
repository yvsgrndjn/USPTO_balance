import os 
import sys
import time
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and modules
from config_create_src_tgt_files import (BASE_DIR, FOLDER_PATH, DATASET_NAME, TARGET_RXNS_PER_TEMPLATE, PATH_SUBSET_SPLIT_CSV)
from src.uspto_balance.data_handler import DataHandler
from src.uspto_balance.create_src_tgt_files import CreateSrcTgtFiles

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create source and target files for the training of each of the models constituting a TTL or TTL* model')

    # Optional arguments to override config settings
    parser.add_argument('--folder-path', type=str, default=FOLDER_PATH, help='Folder from which the enrichment was run')
    parser.add_argument('--dataset-name', type=str, default=DATASET_NAME, help='Name of the dataset')
    parser.add_argument('--path-subset-split-csv', type=str, default=PATH_SUBSET_SPLIT_CSV, help='Path to the csv dataset of all created and validated reactions with a target number of reactions per template')
    parser.add_argument('--target-rxns-per-template', type=int, default=TARGET_RXNS_PER_TEMPLATE, help='Target number of reactions per template in the final dataset')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_arguments()

    # load the split dataframe needed to actually create the source and target files
    datahandler = DataHandler(file_path=args.path_subset_split_csv)
    df = datahandler.load_data()

    #------------------------------------
    # TESTING: only use the first 100 rows
    #df = df.sample(100, random_state=42)
    #------------------------------------

    # get the mapped smiles and splits, delete the dataframe
    mapped_rxns = df['mapped_rxns'].tolist()
    splits = df['Set'].tolist()
    del df

    # initialize the CreateSrcTgtFiles class
    create_src_tgt_files = CreateSrcTgtFiles(mapped_rxns, splits, args.dataset_name, BASE_DIR, args.target_rxns_per_template)

    # create source and target files
    create_src_tgt_files.process_and_save_all()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds')