import os
import sys 
import math
import time
import logging
import argparse
import numpy as np

# Append the parent directory to sys.path to import modules from src

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and modules 
from config_stats_analysis import (FOLDER_PATH, DATASET_NAME, DF_TEMP_PATH, STATS_PATH, FRAC_PATH, PATH_SAVE_STATS)
from src.uspto_balance.data_handler import DataHandler
import src.uspto_balance.balancing_workflow as bw

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and validate fictive reactions for a given template with flexible options.")
    
    # Optional arguments to override config settings
    parser.add_argument('--folder_path', type=str, default=FOLDER_PATH, help="Folder from which the equilibration was run")
    parser.add_argument('--dataset-name', type=str, default=DATASET_NAME, help="Name of the dataset.")
    parser.add_argument('--df-template-path', type=str, default=DF_TEMP_PATH ,help= "Path of the dataframe containing the templates to enrich.")
    parser.add_argument('--stats_path', type=str, default=STATS_PATH, help="Path where the statistics of the different enrichments are saved")
    parser.add_argument('--fraction-path', type=str, default=FRAC_PATH, help="Path to the first fraction of the dataset (used to determine the length of each fraction)")
    parser.add_argument('--path-to-save-stats', type=str, default=PATH_SAVE_STATS, help=" Path to save the final statistics file of the enrichment")
    parser.add_argument('--log', type=bool, default=False, help="Saving logs to output.log file. Default False")
    return parser.parse_args()


def main():
    args = parse_arguments()
    # initialize datahandler
    datahandler = DataHandler(file_path=args.df_template_path)
    # load df_templates
    df_templates = datahandler.data

    # reunite all the stats files into a single dataframe
    df_template_stats = bw.load_stats_csv_into_df_old(args.stats_path, args.dataset_name, df_templates)

    # add the column number_tags_prod from df_templates to the df_template_stats
    df_template_stats['number_tags_prod'] = df_templates['number_tags_prod']

    to_keep, nan_values, timelimit, no_matches, no_val_rxn, no_val_conf_rxn = bw.stats_preprocessing(df_template_stats)
    df_template_stats_filter = df_template_stats[to_keep].copy()
    print('Templates with NaN values to investigate/rerun:', np.arange(0,len(df_template_stats))[nan_values])
    print('Templates with timelimit to rerun:', np.arange(0,len(df_template_stats))[timelimit])

    # calculate the statistics for the created reactions

    # initialize 
    fraction1 = DataHandler(file_path=args.fraction_path)
    fraction_length = fraction1.__len__()
    df_datafractions = df_template_stats_filter['dataset fractions']
    df_mol_match = df_template_stats_filter['molecules match']
    df_created_rxn = df_template_stats_filter['created_rxns']
    df_val_rxn = df_template_stats_filter['validated reactions']
    df_val_conf_rxn = df_template_stats_filter['validated and confident reactions']

    # calculate statistics and add to the dataframe
    df_template_stats_filter['molecules_per_match'], df_template_stats_filter['match_per_created_rxn'], df_template_stats_filter['created_rxn_per_val_rxn'], df_template_stats_filter['val_rxn_per_val_conf_rxn'], df_template_stats_filter['molecules_per_val_conf_rxn'] = bw.calculate_stats_enrichment(fraction_length, df_datafractions, df_mol_match, df_created_rxn, df_val_rxn, df_val_conf_rxn)

    df_template_stats_filter.to_csv(args.path_to_save_stats)

if __name__ == '__main__':
    start_time = time.time() 
    main() 
    end_time = time.time()
    logging.info(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")