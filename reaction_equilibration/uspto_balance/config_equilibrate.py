import os

# =======================
# General Configuration
# =======================

# Base directory of the project
FOLDER_PATH = '/home/yves/Documents/GitHub/test_folder_uspto_balance/'

# =======================
# Enrichment configuration
# =======================
DATASET_NAME        = 'USPTO_rand'
RETRO_REACTANT      = '[C:2]-[OH;D1;+0:1]'
RETRO_TEMPLATE      = '[C:2]-[OH;D1;+0:1]>>C-[O;H0;D2;+0:1]-[C:2]'
TEMPLATE_LINE       = 0
TEMPLATE_FREQUENCY  = 0
FREQUENCY_TARGET    = 2300

# =======================
# Model paths
# =======================
T2_MODEL_PATH = '/home/yves/Documents/GitHub/TTL_versions/1.4/models/USPTO_STEREO_separated_T2_Reagent_Pred_225000.pt'
T3_MODEL_PATH = '/home/yves/Documents/GitHub/TTL_versions/1.4/models/T3_Fwd_Tag_model_step_300000.pt'

# =======================
# Logging Configuration
# =======================

# ..

# =======================
# Other Configurations
# =======================

# Number of CPU cores to use for parallel processing
N_JOBS = os.cpu_count()  # Uses all available cores

# Random seed for reproducibility
RANDOM_STATE = 42