# uspto_balance

[![License](https://img.shields.io/pypi/l/uspto_balance.svg?color=green)](https://github.com/yvsgrndjn/uspto_balance/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/uspto_balance.svg?color=green)](https://pypi.org/project/uspto_balance)
[![Python Version](https://img.shields.io/pypi/pyversions/uspto_balance.svg?color=green)](https://python.org)
[![CI](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml/badge.svg)](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml)

Module to create similar reactions to a given retrosynthetic reaction template. First, the reaction template is being applied to molecules containing the same substructure as the product side. Second, each reaction will have its reagent(s) predicted before being validated by a disconnection-aware forward validation model. A reaction is considered valid if the original target product is found back during the forward validation with a confidence score higher than 95%.

## Project structure
~~~
USPTO_balance/ 
├── data/ 
│   ├── dataframes/
│   │   ├── {dataset_name}/
│   │   │      ├──df_templates.pkl 
│   │  ..   ..
│   ├── datasets/
│   │   └── {dataset_name}/
│   │ 		 ├── {dataset_name}_i.txt (i = {1,…,n})		# dataset split into n=1000 subsets
│   │ ..		 └── {dataset_name}_i_mol.pkl (i = {1,…,n}) 	# molecules from dataset split converted to rdkit mol-object
│   ├── models/ 
│   │   ├── T1/..
│   │   ├── T2/..
│   │   ├── T3/..
│   │   └── T3FT..
│
├── src/uspto_balance/ 
│   ├── __init__.py 
│   ├── data_handler.py           		 # Contains DataHandler class 
│   ├── find_matches_and_format.py	 # class containing the process to find matches and format reactions 
│   ├── validate_reactions_TTL.py	 # validates reactions with the TTL 
│   ├── equilibrate.py			 # processes the whole enrichment process 
│   ├── stats_analysis.py
│   ├── build_dataset.py 
│   ├── create_src_tgt_files.py
│   ├── balancing_workflow.py		 # utilities that should be cleaned into something shorter 
│   ├── execute_onmt.py			 # utilities that should be cleaned into something shorter 
│   └── utils/ 
│       ├── __init__.py 
│       └── helper_functions.py   		 # Any additional helper functions
│ 
├── notebooks/ 
│   └── exploratory_analysis.ipynb	 # Jupyter notebooks for testing and analysis 
│
├── tests/ 
│   ├── __init__.py 
│  ..
│
├── scripts/ 
│   ├── run_equilibrate.py           		 # Script to execute the entire pipeline 
│   ├── run_stats_analysis.py     
│   ├── run_build_dataset.py 
│   └── run_create_src_tgt_files.py
│
├──  pyproject.toml          		  
├── README.md                      
├── .gitignore                     
├── config_equilibrate.py 
├── config_stats_analysis.py                     
├── config_build_dataset.py
├── config_create_src_tgt_files.py
└── LICENSE                        
~~~

## Installing

Once the `git clone` performed, ensure you are in the `USPTO_balance/reaction_equilibration/uspto_balance` folder. To create the environment:
~~~
conda create -n uspto_balance python=3.8.16
conda activate uspto_balance
pip install -e .
~~~

## Adding the necessary files to run enrichment
~~~
mkdir data/
cd data
mkdir dataframes datasets models
~~~
### Dataframe containing the templates to enrich
Save the dataframe containing the retro templates under `data/dataframes/{dataset_name}/`

### Datasets 
All the available molecules to apply the templates on are our pool of molecules. The calculations are performed on subsets of the total pool. For a given `{dataset_name}`, split it into 1000 subsets with names `{dataset_name}_i.txt` for i = {1, .., n} (n=1000 by default). The same splits with all molecules converted to rdkit mol objects must also be present under generic name `{dataset_name}_i_mol.pkl`. 
These 2000 files must be saved under `data/datasets/{dataset_name}` folder.

### Models
download the T2 and T3* (called T3FT) models trained on USPTO from zenodo: https://zenodo.org/records/14017743?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY2NmZmNTkxLTE2YzYtNDU0OS04NjAzLTJiNzg1YzFhMGQ5NSIsImRhdGEiOnt9LCJyYW5kb20iOiJmMjBhNjlkNDRkYTE1YWMwNDVjODQ2YjkwOTQ1ZjgyNCJ9.DRQXQBjRcv6MTW1hEYDSYZ6j11dmKBAQI-nytHBTHKu66KYTS3TgriJW_pOTfayHcditLS4MNKa9okI4FLSD2Q

save the models under `data/models/{model}/USPTO_IBM/`

## Enrich a retrosynthetic template

### Prepare config_equilibrate.py
Check that the default arguments of the config file (mostly the paths) correspond to your case:
```
#--config_equilibrate.py--

import os

# =======================
# General Configuration
# =======================

# Base directory of the project
FOLDER_PATH = './'							# from where the code will be run

# =======================
# Enrichment configuration
# =======================
DATASET_NAME        = 'USPTO_rand'
RETRO_REACTANT      = '[C:2]-[OH;D1;+0:1]' 				# left side of retro_template
RETRO_TEMPLATE      = '[C:2]-[OH;D1;+0:1]>>C-[O;H0;D2;+0:1]-[C:2]'	# retrosynthetic template to enrich
TEMPLATE_LINE       = 0							# line at which the template is found in the df_templates
TEMPLATE_FREQUENCY  = 0							# number of reactions at the beginning of the encrichment
FREQUENCY_TARGET    = 5000						# target number of reactions

# =======================
# Model paths
# =======================
T2_MODEL_PATH = './data/models/T2/USPTO_IBM/USPTO_T2.pt
T3_MODEL_PATH = './data/models/T3FT/USPTO_IBM/USPTO_T3FT.pt

# =======================
# Other Configurations
# =======================

# Number of CPU cores to use for parallel processing
N_JOBS = os.cpu_count()  # Uses all available cores

# Random seed for reproducibility
RANDOM_STATE = 42
```

### Run enrichment
The config file values are there as defaults, they can be modified to be adapted to each use case. To run the enrichment with different arguments, enter following command from `uspto_balance/` with the `uspto_balance` environment activated:
```
python scripts/run_equilibrate.py \
        --folder_path "${FOLDER_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --retro-reac "${RETRO_REACTANT}" \
        --retro-template "${RETRO_TEMPLATE}" \
        --template-line "${TEMPLATE_LINE}" \
        --frequency-target "${FREQUENCY_TARGET}" \
        --T2-model-path "${T2_MODEL_PATH}" \
        --T3-model-path "${T3_MODEL_PATH}" \
        --n-jobs "${N_JOBS}"
```

...
