import os
import random
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


def convert_and_save_subset(subset, subset_mol, retro_reac, GDB_version: str = '', template_version: str = ''):
    '''
    Saves a subset of SMILES strings to a txt file and converts it to mol before saving it to a pkl file
    '''
    if subset:
        name = f'GDB13S_sub_{retro_reac}'
        folder_path = f'./GDB_subsets_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in subset:
                f.write(item + '\n')

        folder_path_mol = f'./GDB_subsets_mol_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path_mol):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path_mol)
        with open(f'{folder_path_mol}/{name}.pkl', 'wb') as f:
            pickle.dump(subset_mol, f)


def canonicalize(smiles):
    '''
    Converts a smile string into a rdkit canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def extract_match_smiles_from_dataset(dataset:list, dataset_mol:list, template:str):
    """
    This function extracts the elements from a smiles dataset that match a certain template and canonicalizes them
    """
    template_mol    = Chem.MolFromSmarts(template)
    match_ind       = [i for i in range(len(dataset_mol)) if dataset_mol[i].HasSubstructMatch(template_mol)]
    dataset_sub     = [canonicalize(dataset[i]) for i in range(len(dataset)) if i in match_ind]
    dataset_sub_mol = [dataset_mol[i] for i in range(len(dataset_mol)) if i in match_ind]
    return dataset_sub, dataset_sub_mol


#def extract_match_smiles_from_dataset_old(dataset:list, dataset_mol:list, template:str):
#    """
#    This function extracts the elements from a smiles dataset that match a certain template and canonicalizes them
#    """
#    template_mol  = Chem.MolFromSmarts(template)
#    match_ind     = [mol.HasSubstructMatch(template_mol) for mol in dataset_mol]
#    return [canonicalize(dataset[i]) for i in range(len(dataset)) if match_ind[i] == True], match_ind


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(GDB13S_path, df_templates_path_to_pkl, GDB_version, template_version):

    # Load GDB13S dataset
    with open(GDB13S_path, 'r') as f:
        GDB13S = [line.strip() for line in f]

    # Convert SMILES to RDKit molecule objects in parallel
    processes = os.cpu_count() - 2
    with Pool(processes) as p:
        output = list(tqdm(p.imap(smiles_to_mol, GDB13S), total=len(GDB13S)))
    GDB13S_mol = output

    #load df_templates prepared in (*)
    df_templates = pd.read_pickle(df_templates_path_to_pkl)

    # Extract subsets based on unique retro_reac values
    unique_retro_reac_values = df_templates['retro_reac'].unique()
    for retro_reac in unique_retro_reac_values:
        GDB13S_sub, GDB13S_sub_mol = extract_match_smiles_from_dataset(GDB13S, GDB13S_mol, retro_reac)
        convert_and_save_subset(GDB13S_sub, GDB13S_sub_mol, retro_reac, GDB_version, template_version)

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
        config['GDB13S_path'],
        config['df_templates_path_to_pkl'],
        config['GDB_version'],
        config['template_version']
    )
    '''config_file = "config_part1.yaml"
    config = read_config(args.config)
    main(
        config['GDB13S_path'],
        config['df_templates_path_to_pkl'],
        config['GDB_version'],
        config['template_version']
    )'''