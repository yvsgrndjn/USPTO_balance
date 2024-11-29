import os

# =======================
# General Configuration
# =======================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory from which the enrichment was run
FOLDER_PATH     = '/home/yves/Documents/GitHub/test_folder_uspto_balance/' # NOT USED FOR NOW

# Name of the dataset used for the enrichment
DATASET_NAME    = 'USPTO_rand'

# Path where the new reactions' dataset will be saved
#PATH_FULL_CSV    = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_created_validated_reactions.csv')

# Path where the original dataset is saved
#PATH_TO_REF_DATA = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_Thakkar_12_2023_Mapped_template_info_r0r1_full_corr_hash.pkl')

# Target number of reactions per template in the final dataset (only for naming files, the dataset has been built already)
TARGET_RXNS_PER_TEMPLATE = 100

# Path to save the splitted subset 
PATH_SUBSET_SPLIT_CSV = os.path.join(BASE_DIR, 'data/dataframes', DATASET_NAME, 'df_created_validated_'+ str(TARGET_RXNS_PER_TEMPLATE)+'_reactions_per_template_split.csv')

TEMPLATE_COLUMN_NAME = 'retro_template'
# Proportion of training reactions in the final dataset split
#TRAIN_SET_PROP = 0.8
