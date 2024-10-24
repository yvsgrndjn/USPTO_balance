import os

# =======================
# General Configuration
# =======================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory from which the enrichment was run
FOLDER_PATH = '/home/yves/Documents/GitHub/test_folder_uspto_balance/'

# Name of the dataset used for the enrichment
DATASET_NAME        = 'USPTO_rand'

# Path where the dataframe containing the templates to enrich is saved
DF_TEMP_PATH        = BASE_DIR + '/data/dataframes/' + DATASET_NAME + '/df_templates_to_enrich_templates_r0r1_numtags.pkl'

# Path where the statistics of the different enrichments are saved
STATS_PATH          = FOLDER_PATH + '/results/saved_rxns/' + DATASET_NAME + '/'

# Path to the first fraction of the dataset (used to determine the length of each fraction)
FRAC_PATH           = BASE_DIR + '/data/' + DATASET_NAME + '/' + DATASET_NAME + '_1.txt'

# Path to save the final statistics file of the enrichment
PATH_SAVE_STATS    = BASE_DIR + '/data/dataframes/' + DATASET_NAME + '/df_template_stats_filtered.csv'
