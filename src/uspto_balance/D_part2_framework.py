import os
import pickle
from itertools import chain
import pandas as pd
from tqdm import tqdm
import yaml
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import argparse
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


def load_subsets(retro_reac: str = '', dataset_version: str = '', template_version: str = '', dataset_name: str = ''): 
    '''
    Loads both SMILES and mol versions that were calculated in part 1 of the dataset subset matching the retro_reac pattern. 
    dataset version is the subset's number
    '''
    
    folder_path     = f'./results/datasets/{dataset_name}'
    folder_path_mol = f'./results/datasets/{dataset_name}_mol'
    name            = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'

    try:
        with open(f'{folder_path}/{name}.txt', 'r') as f:
            dataset_sub = [line.strip() for line in f]
        with open(f'{folder_path_mol}/{name}.pkl', 'rb') as f:
            dataset_sub_mol = pickle.load(f)

        return dataset_sub, dataset_sub_mol

    except FileNotFoundError:
        print(f'No subsets found for retro_reac: {retro_reac} in dataset version: {dataset_version}')
        return [], []


def save_rxns(rxns_list, retro_reac, retro_template, dataset_version: str = '', template_version: str = '', dataset_name: str = ''):
    '''
    Saves the rxn list in a txt file, retro_reac is the SMARTS pattern of the product of the reaction,
    retro_template is the template that is applied on retro_reac
    '''

    #Remove the slash from the template
    retro_template = retro_template.replace('/', 'slash')

    if rxns_list:
        folder_path = f'./results/created_rxns/{dataset_name}'
        name = f'rxns_{dataset_version}_{retro_reac}_{retro_template}'

        #Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in rxns_list:
                f.write(item + '\n')
        print(f'Created {len(rxns_list)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')


def apply_rxn_template_on_mols_list(dataset_mol:list, rxn_template:str):
    '''
    Applies a reaction template on a list of mols (dataset) and returns a list of lists of reactants

    dataset: list of molecules mols 
    rxn_template: reaction template in string format
    ---
    returns: list of lists of reactants
    '''
    rxn = AllChem.ReactionFromSmarts(rxn_template)

    return [rxn.RunReactants((dataset_mol[i],)) for i in range(len(dataset_mol))]


def canonicalize(smiles :str):
    '''
    Converts a smile string into a rdkit canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def format_reaction(reactants_tuple: tuple, smi : str):
    '''From the runreactants result, returns the reactions in a smiles format'''

    reactants_smiles_list = []
    for i in range(len(reactants_tuple)):
        reactants_mol = list(reactants_tuple[i])

        reactants_smiles = ''
        #reactants_smiles_list = []
        for j in range(len(reactants_mol)):
            reactants_smiles += Chem.MolToSmiles(reactants_mol[j]) + '.'
        reactants_smiles = reactants_smiles[:-1]
        reactants_smiles_list.append(reactants_smiles)
    
    reactants_smiles_list = list(set(reactants_smiles_list))
    rxn = [canonicalize(reactants_smiles_list[i]) + '>>' + smi for i in range(len(reactants_smiles_list))]
    
    return rxn


def process_retro_template(retro_reac, retro_template, dataset_version: str = '', template_version: str = '', dataset_name: str = ''):
    
    dataset_sub, dataset_sub_mol = load_subsets(retro_reac, dataset_version, template_version, dataset_name)

    if not dataset_sub:
        return

    # Apply template
    dataset_sub_app_temp = apply_rxn_template_on_mols_list(dataset_sub_mol, retro_template)

    # Find indices of empty elements and remove them from both lists
    ind_remove = [result == () for result in dataset_sub_app_temp]
    dataset_sub_app_temp_sort = [dataset_sub_app_temp[i] for i in range(len(dataset_sub_app_temp)) if not ind_remove[i]]
    dataset_sub_sort = [dataset_sub[i] for i in range(len(dataset_sub)) if not ind_remove[i]]

    # Create fictive reactions
    fictive_rxns_list = [format_reaction(dataset_sub_app_temp_sort[k], dataset_sub_sort[k]) for k in range(len(dataset_sub_sort))]
    fictive_rxns_list = list(chain.from_iterable(fictive_rxns_list))

    # Save in a txt file
    save_rxns(fictive_rxns_list, retro_reac, retro_template, dataset_version, template_version, dataset_name)


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(dataset_name, dataset_version, template_version, retro_reac, retro_template):
    
    process_retro_template(retro_reac, retro_template, dataset_version, template_version, dataset_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to the configuration file')
    args = parser.parse_args()

    if not args.config:
        print('Please provide a configuration file')
        sys.exit()
    elif not os.path.exists(args.config):
        print('The configuration file does not exist')
        sys.exit()

    config = read_config(args.config)
    main(
        config['dataset_name'],
        config['dataset_version'],
        config['template_version'],
        config['retro_reac'],
        config['retro_template']
<<<<<<< HEAD
    )
=======
    )
>>>>>>> 35f4b8cbaaf8e23a4df52e329f61332a24d10264
