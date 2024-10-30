import os

# =======================
# General Configuration
# =======================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory from which the enrichment was run
FOLDER_PATH     = '/home/yves/Documents/GitHub/test_folder_uspto_balance/'

# Name of the dataset used for the enrichment
DATASET_NAME    = 'USPTO_rand'

# Path where the dataframe containing the templates to enrich is saved
DF_TEMP_PATH    = os.path.join(BASE_DIR, 'data/dataframes',  DATASET_NAME, 'df_templates_to_enrich_templates_r0r1_numtags.pkl')

# Path where the new reactions' dataset will be saved
PATH_TO_SAVE_FULL_CSV    = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_created_validated_reactions.csv')

# Path where the original dataset is saved
PATH_TO_REF_DATA = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_Thakkar_12_2023_Mapped_template_info_r0r1_full_corr_hash.pkl')

# Target number of reactions per template in the final dataset
TARGET_RXNS_PER_TEMPLATE = 100

# Path to save the splitted subset 
PATH_TO_SAVE_SUBSET_SPLIT = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_created_validated_'+ str(TARGET_RXNS_PER_TEMPLATE)+'_reactions_per_template_split.csv')

# Proportion of training reactions in the final dataset split
TRAIN_SET_PROP = 0.8
