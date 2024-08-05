# uspto_balance

[![License](https://img.shields.io/pypi/l/uspto_balance.svg?color=green)](https://github.com/yvsgrndjn/uspto_balance/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/uspto_balance.svg?color=green)](https://pypi.org/project/uspto_balance)
[![Python Version](https://img.shields.io/pypi/pyversions/uspto_balance.svg?color=green)](https://python.org)
[![CI](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml/badge.svg)](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml)

**This repository is a work in progress**

Module creating similar reactions to a given retrosynthetic reaction template using a pool of molecules. First, the reaction template is being applied to molecules containing the same substructure as the product side. Second, each reaction will have its reagent(s) predicted before being validated by a disconnection-aware forward validation model. A reaction is considered valid if the original target product is found back during the forward validation with a confidence score higher than 95%.

## Installing
### From GitHub
~~~
git clone git@github.com:yvsgrndjn/USPTO_balance.git
cd uspto_balance
conda create uspto_balance
conda activate uspto_balance
pip install -e .
~~~

## Usage
### Prerequisites
Folder structure must look like this, where *dataset_name* can be any name. Under datasets/dataset_name/ the pool of that will be used throughout the enrichment must be split in 1000 separate txt files (can be all the molecules of USPTO separated into 1000 files)

~~~
folder_to_run_from/
├── results/
│   ├── datasets/
│   │   └── dataset_name/
│   │   	├── dataset_name_1.txt
│   │   	├── dataset_name_2.txt
│   │   	├ ...
│   │   	├── dataset_name_1000.txt
│   ├── created_rxns/
│   │   └── dataset_name/
│   ├── saved_rxns/
│   │   └── dataset_name/
│   ├── temp_files/
│   ├── models/ #store here the retrosynthesis models from zenodo
~~~

### Running the code
cd to the folder you want to run the method from, with the uspto_balance environment activated:
~~~
balance path/to/config.yaml
~~~

the config file must look like (YAML):
~~~
dataset_name: 'USPTO' #used for folder names
retro_reac: '[#7;a:10]:[c;H0;D3;+0:9](:[c:11])-[c;H0;D3;+0:1]1:[n;H0;D2;+0:12]:[c:13]:[n;H0;D3;+0:14](:[c:15]):[c;H0;D3;+0:2]:1-[c;H0;D3;+0:3]1:[c:4]:[c:5]:[#7;a:6]:[c:7]:[c:8]:1' #reactant of the template
retro_template: '[#7;a:10]:[c;H0;D3;+0:9](:[c:11])-[c;H0;D3;+0:1]1:[n;H0;D2;+0:12]:[c:13]:[n;H0;D3;+0:14](:[c:15]):[c;H0;D3;+0:2]:1-[c;H0;D3;+0:3]1:[c:4]:[c:5]:[#7;a:6]:[c:7]:[c:8]:1>>O=[C;H0;D3;+0:1](-[CH2;D2;+0:2]-[c;H0;D3;+0:3]1:[c:4]:[c:5]:[#7;a:6]:[c:7]:[c:8]:1)-[c;H0;D3;+0:9](:[#7;a:10]):[c:11].[NH2;D1;+0:12]-[c:13]:[n;H0;D2;+0:14]:[c:15]' #full template to enrich
template_hash: '58a1fbc8d3c1fcce803afb4bf25ac6fa0ae628520b49d64465397906' #used for file names
template_line: 14324 #used for file names
path_to_folder: './test_folder/' #path leading to where the method is run from
path_models: 'path/to/models/' #path to the retrosynthesis models are stored
template_frequency: 0 #actual number of reactions following the template reactivity
frequency_target: 5000 #enrichment target 
~~~


