import os
import sys
import time
import logging
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and modules 
from config_equilibrate import (FOLDER_PATH, DATASET_NAME, RETRO_REACTANT, RETRO_TEMPLATE, TEMPLATE_LINE, 
                    TEMPLATE_FREQUENCY, FREQUENCY_TARGET, T2_MODEL_PATH, T3_MODEL_PATH, 
                    N_JOBS, RANDOM_STATE)
from src.uspto_balance.equilibrate import Equilibrate

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and validate fictive reactions for a given template with flexible options.")
    
    # Optional arguments to override config settings
    parser.add_argument('--folder_path', type=str, default=FOLDER_PATH, help="Folder from which to run the equilibration.")
    parser.add_argument('--dataset-name', type=str, default=DATASET_NAME, help="Name of the dataset.")
    parser.add_argument('--retro-reac', type=str, default=RETRO_REACTANT ,help= "reactant side of the retrosynthetic template to enrich")
    parser.add_argument('--retro-template', type=str, default=RETRO_TEMPLATE, help="Retrosynthetic template to enrich.")
    parser.add_argument('--template-line', type=int, default=TEMPLATE_LINE, help="Line number of the template in the dataset.")
    parser.add_argument('--template-frequency', type=int, default=TEMPLATE_FREQUENCY, help="Frequency of the template in the dataset.")
    parser.add_argument('--frequency-target', type=int, default=FREQUENCY_TARGET, help="Target frequency for the template enrichment")
    parser.add_argument('--T2-model-path', type=str, default=T2_MODEL_PATH, help="Path to the T2 model.")
    parser.add_argument('--T3-model-path', type=str, default=T3_MODEL_PATH, help="Path to the T3 model.")
    parser.add_argument('--n-jobs', type=int, default=N_JOBS, help="Number of CPU cores to use.")
    parser.add_argument('--log', type=bool, default=False, help="Saving logs to output.log file. Default False")
    return parser.parse_args()


def main():
    args = parse_arguments()
    equilibrate_dataset = Equilibrate(args.dataset_name, 
                                      args.retro_reac, 
                                      args.retro_template, 
                                      args.template_line, 
                                      args.template_frequency, 
                                      args.frequency_target, 
                                      args.T2_model_path, 
                                      args.T3_model_path, 
                                      args.folder_path
                                      )
    equilibrate_dataset.process()

if __name__ == '__main__':
    start_time = time.time() 
    main() 
    end_time = time.time()
    logging.info(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")

