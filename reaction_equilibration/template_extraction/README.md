# Template_extraction

this package extracts radius 0 and 1 templates from a dataframe of mapped reactions.
The dataframe must contain the mapped reactions in a column named 'MAPPED_SMILES', the reactions are then transformed in an output allowing the template extraction to be pursued and ensure that the outputs are working as expected.
The resulting dataframe is saved as a csv and can be directly used as input for the template correction.

## Installation

```
cd reaction_equilibration/template_extraction
conda create --name template_extraction python=3.9 -y
conda activate template_extraction
```

## Format input and extract templates
To perform calculations from a jupyter notebook for example:
```
from template_extraction.pipeline import process_and_extract_templates

input_csv_path = './tests/uspto_Thakkar_sample_1000_mapped_smiles.csv'
output_csv_path = './tests/uspto_Thakkar_sample_1000_r0r1.csv'

process_and_extract_templates(input_csv_path, output_csv_path)
```

The results of the template extraction are available in the ```output_csv_path```

Note that for bigger datasets it is useful to run this with multiprocess to accelerate calculations

## Extract and correct templates from a file
from https://github.com/hesther/templatecorr, follow installation of the package

Once templatecorr environment is activated:
```
import pickle
import pandas as pd
import templatecorr
from rdkit import Chem
from tqdm import tqdm
import os

output_csv_path = './tests/uspto_Thakkar_sample_1000_r0r1.csv'
df = pd.read_csv(output_csv_path)

# Correct templates
nproc=14
templatesAZ_corr_V1 = templatecorr.correct_all_templates(df, 'template_r0', 'template_r1', nproc)

#rename the column containing the corrected R1 templates
df.rename(columns={'new_t':'template_r1_corr'}, inplace=True)
df.to_csv(output_csv_path)
```


