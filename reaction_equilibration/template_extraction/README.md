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

To perform calculations from a jupyter notebook for example:
```
from template_extraction.pipeline import process_and_extract_templates

input_csv_path = './tests/uspto_Thakkar_sample_1000_mapped_smiles.csv'
output_csv_path = './tests/uspto_Thakkar_sample_1000_r0r1.csv'

process_and_extract_templates(input_csv_path, output_csv_path)
```

The results of the template extraction are available in the ```output_csv_path```

Note that for bigger datasets it is useful to run this with multiprocess to accelerate calculations
