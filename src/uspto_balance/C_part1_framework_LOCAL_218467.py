import os
import random
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import sys
import argparse
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


def smiles_to_mol(smi):
    '''
    Converts a smile string into a rdkit mol file
    '''
    mol = Chem.MolFromSmiles(smi)
    return mol


def do_subsets_exist_already(dataset_name: str, dataset_version: str, retro_reac: str):
    '''
    Checks if the subsets matching a retro_reac pattern in the dataset version have already been extracted.
    Returns True if they exist, False otherwise

    Inputs:
    dataset_name: name of the dataset (str) ex: GDB13S, USPTO. Prerequisite: The dataset divided in 100 different files in the format {dataset_name}_i.txt for i from 1 to 100 must be present in the folder dataset_balance/data/
    dataset_version: version of the dataset (str) being any integer from 1 to 100
    retro_reac: SMARTS pattern (str) of the retrosynthetic precursor of the reaction. ex: output of the rxn_smarts_to_sanitized_reactant_smarts function
    '''
    folder_path     = f'./results/datasets/{dataset_name}'
    folder_path_mol = f'./results/datasets/{dataset_name}_mol'
    name            = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'    

    if os.path.exists(f'{folder_path}/{name}.txt') and os.path.exists(f'{folder_path_mol}/{name}.pkl'):
        return True
    else:
        return False


def convert_and_save_subset(subset, subset_mol, dataset_name:str, retro_reac, dataset_version: str = '', template_version: str = ''):
    '''
    Saves a subset of SMILES strings to a txt file and converts it to mol before saving it to a pkl file
    '''
    if subset:
        folder_path = f'./results/datasets/{dataset_name}'
        name        = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'

        #Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #Write the subset to a text file    
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in subset:
                f.write(item + '\n')


        folder_path_mol = f'./results/datasets/{dataset_name}_mol'

        #Create the folder if it does not exist
        if not os.path.exists(folder_path_mol):
            os.makedirs(folder_path_mol)
        
        #Save the subset to a pickle file
        with open(f'{folder_path_mol}/{name}.pkl', 'wb') as f:
            pickle.dump(subset_mol, f)

        print(f'Saved subset of {len(subset)} smiles from {dataset_name}_{dataset_version} for retro_reac: {retro_reac}')


def canonicalize(smiles):
    '''
    Converts a smile string into a rdkit canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def extract_match_smiles_from_dataset(dataset:list, dataset_mol:list, template:str):
    """
    This function extracts the elements from a smiles dataset that match a certain template and canonicalizes them
    """
    #convert template to mol
    template_mol    = Chem.MolFromSmarts(template)

    #find indices in the dataset of the substructure matches
    match_ind = [i for i, mol in enumerate(dataset_mol) if mol.HasSubstructMatch(template_mol)]

    #create a subset of canonicalized smiles containing the substructure matches
    dataset_match = [dataset[i] for i in match_ind]
    processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=processes)
    dataset_sub = pool.map(canonicalize, dataset_match)
    pool.close()
    pool.join()

    #create subset of mol objects containing the substructure matches
    dataset_sub_mol = [dataset_mol[i] for i in match_ind]
    return dataset_sub, dataset_sub_mol



def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(dataset_name, dataset_path, dataset_version, template_version, retro_reac, retro_template):

    if do_subsets_exist_already(dataset_name, dataset_version, retro_reac):
        print(f'The subsets for dataset {dataset_name} and retro_reac {retro_reac} already exist')
        pass

    else:
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = [line.strip() for line in f]

        # Convert SMILES to RDKit mol objects
        processes = os.cpu_count() - 2
        with Pool(processes) as p:
            output = list(tqdm(p.imap(smiles_to_mol, dataset), total=len(dataset)))
        dataset_mol = output

        dataset_sub, dataset_sub_mol = extract_match_smiles_from_dataset(dataset, dataset_mol, retro_reac)
        convert_and_save_subset(dataset_sub, dataset_sub_mol, dataset_name, retro_reac, dataset_version, template_version)


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
        config['dataset_path'],
        config['dataset_version'],
        config['template_version'],
        config['retro_reac'],
        config['retro_template']
    )