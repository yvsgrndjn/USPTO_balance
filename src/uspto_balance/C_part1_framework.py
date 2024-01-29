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
import pandas as pd
import argparse
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


def smiles_to_mol(smi: str) -> Chem.Mol:
    '''
    Converts a smile string into a rdkit mol file

    Inputs:
    smi: smile string (str)

    Returns:
    mol: rdkit mol object
    '''
    mol = Chem.MolFromSmiles(smi)
    return mol


def do_subsets_exist_already(dataset_name: str, dataset_version: str, template_hash_version: str) -> bool:
    '''
    Checks if the subsets containing the SMILES/mol-format molecules containing the 'retro_reac' substructure already exist (= have already been extracted).

    --Inputs--
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 100 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 100 must be present in the folder dataset_balance/data/
    dataset_version (str):          version of the dataset (str) being any integer from 1 to 100
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows.
                                    template_hash_version = f"{template_hash}_{template_line}"

    --Returns--
    (bool) True if they exist, False otherwise
    '''
    # The paths and names of the files and folders are created in the same way throughout the whole module. 
    folder_path     = f'./results/datasets/{dataset_name}'
    folder_path_mol = f'./results/datasets/{dataset_name}_mol'
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'        

    #Check for existence of the files in the given paths
    if os.path.exists(f'{folder_path}/{name}.txt') and os.path.exists(f'{folder_path_mol}/{name}.pkl'):
        return True
    else:
        return False


def canonicalize(smiles: str) -> str:
    '''
    Converts a smile string into a rdkit canonicalized smile string

    --Inputs--
    smiles (str):   smile string to canonicalize

    --Returns--
    (str) canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def extract_match_smiles_from_dataset(dataset:list, dataset_mol:list, template:str) -> list:
    """
    Creates subsets from a smiles dataset (and Chem.mol dataset) with canonicalized elements matching a certain SMARTS pattern template.

    --Inputs--
    dataset (list(str)):             list of smiles strings to extract the molecules matching the template from
    dataset_mol (list(Chem.Mol)):    list of mol objects corresponding to the smiles strings
    template (str):                  SMARTS pattern of the substructure to match

    --Returns--
    dataset_sub:        list of smiles strings containing the substructure matches. Subset of the input dataset.
    dataset_sub_mol:    list of mol objects corresponding to the smiles strings containing the substructure matches. Subset of the input dataset_mol.
    """
    # Convert template to mol
    template_mol    = Chem.MolFromSmarts(template)

    # Find indices of the substructure matches in the dataset
    match_ind = [i for i, mol in enumerate(dataset_mol) if mol.HasSubstructMatch(template_mol)]

    # Create a subset of canonicalized smiles containing the substructure matches 
    dataset_match = [dataset[i] for i in match_ind]
    processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=processes)
    dataset_sub = pool.map(canonicalize, dataset_match)
    pool.close()
    pool.join()

    # Create subset of mol objects containing the substructure matches
    dataset_sub_mol = [dataset_mol[i] for i in match_ind]
    return dataset_sub, dataset_sub_mol


def save_created_files_to_temp_file(temp_path: str, temp_name: str, temp_list: list):
    '''
    Saves the paths of the created files in a temp file to delete them once they are no longer needed at the end of the dataset_version iteration

    --Inputs--
    temp_path (str):    path to the temp folder
    temp_name (str):    name of the temp file
    temp_list (list):   list of paths to the created files

    --Returns--
    None, but creates a txt file containing the paths of the created files that will be deleted at the end of the dataset_version iteration
    '''
    # Create the folder if it does not exist
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
               
    # Create the temp file for the given dataset_name, and template_hash_version
    if not os.path.exists(f'{temp_path}/{temp_name}.txt'):
        with open(f'{temp_path}/{temp_name}.txt', 'w') as f:
            for item in temp_list:
                f.write(item + '\n')
    else:
        with open(f'{temp_path}/{temp_name}.txt', 'a') as f:
            for item in temp_list:
                f.write(item + '\n')


def convert_and_save_subset(subset: list, subset_mol: list, dataset_name:str, retro_reac, dataset_version: str = '', template_hash_version: str = ''): 
    '''
    Saves the SMILES and Chem.mol subsets to a txt and to a pickle file, respectively.

    --Inputs--
    subset (list(str)):             list of SMILES strings to save
    subset_mol (list(Chem.Mol)):    list of Chem.mol objects (same as subset but in mol format)
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 100 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 100 must be present in the folder dataset_balance/data/
    dataset_version (str):          version of the dataset (str) being any integer from 1 to 100
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows. 

    --Returns--
    None, but saves the subsets to f'./results/datasets/{dataset_name}/{dataset_name}_sub_{dataset_version}_{template_hash_version}.txt' (smiles subset) and
    f'./results/datasets/{dataset_name}_mol/{dataset_name}_sub_{dataset_version}_{template_hash_version}.pkl' (mol subset)
    '''
    # Check for subset existence
    if subset:

        # Define paths and names
        folder_path     = f'./results/datasets/{dataset_name}'
        folder_path_mol = f'./results/datasets/{dataset_name}_mol'
        name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

        temp_path       = f'./results/temp_files/{dataset_name}_temp'
        temp_name       = f'{dataset_name}_temp_{template_hash_version}'
        temp_list       = []
        
        #Create the folder if it does not exist (SMILES)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #Write the subset to a text file (SMILES)
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in subset:
                f.write(item + '\n')

            temp_list.append(f'{folder_path}/{name}.txt')

        #Create the folder if it does not exist (Chem.Mol)
        if not os.path.exists(folder_path_mol):
            os.makedirs(folder_path_mol)
        
        #Save the subset to a pickle file (Chem.Mol)
        with open(f'{folder_path_mol}/{name}.pkl', 'wb') as f:
            pickle.dump(subset_mol, f)

            temp_list.append(f'{folder_path_mol}/{name}.pkl')
        
        #Save the paths of saved subsets to a temp file to delete them once they are no longer needed
        save_created_files_to_temp_file(temp_path, temp_name, temp_list)
        print(f'Saved subset of {len(subset)} smiles from {dataset_name}_{dataset_version} for retro_reac: {retro_reac}')


def read_config(config_file: str) -> dict:
    '''
    Reads the yaml config_file to extract the arguments for the main function

    --Inputs--
    config_file (str): path to the yaml config file

    --Returns--
    config (dict): dictionary containing the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(dataset_name: str, dataset_path: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template):
    '''
    This module is used to create subsets of a dataset composed only of molecules containing a certain substructure (retro_reac). 

    --Inputs--
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 100 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 100 must be present in the folder dataset_balance/data/
    dataset_path (str):             path to the dataset (str) ex: dataset_balance/data/GDB13S_1.txt
    dataset_version (str):          version of the dataset (str) being any integer from 1 to 100.
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows. 
                                    template_hash_version = f"{template_hash}_{template_line}"
    retro_reac (str):               SMARTS pattern of the substructure to match
    retro_template (str):           SMARTS pattern of the template to apply on the molecules later on

    --Returns--
    None, but saves the subsets to f'./results/datasets/{dataset_name}/{dataset_name}_sub_{dataset_version}_{template_hash_version}.txt' (smiles subset) and 
    f'./results/datasets/{dataset_name}_mol/{dataset_name}_sub_{dataset_version}_{template_hash_version}.pkl' (mol subset)
    '''
    # Check for subset existence, if it already exists, skip the whole process
    if do_subsets_exist_already(dataset_name, dataset_version, template_hash_version):
        print(f'The subsets for dataset {dataset_name} and template_hash_version {template_hash_version} already exist')
        pass
        return 0
    else:
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = [line.strip() for line in f]

        # Convert SMILES to RDKit mol objects
        processes = os.cpu_count() - 2
        with Pool(processes) as p:
            output = list(tqdm(p.imap(smiles_to_mol, dataset), total=len(dataset)))
        dataset_mol = output

        # Remove None values from the dataset
        df = pd.DataFrame(data=dataset, columns=['smiles'])
        df['mol'] = dataset_mol
        df = df[[not el for el in df['mol'].isnull().values]]

        # Extract the subset of molecules containing the substructure and save it
        dataset_sub, dataset_sub_mol = extract_match_smiles_from_dataset(dataset=df['smiles'], dataset_mol=df['mol'], template = retro_reac)
        convert_and_save_subset(dataset_sub, dataset_sub_mol, dataset_name, retro_reac, dataset_version, template_hash_version)
        return len(dataset_sub)

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
        config['template_hash_version'],
        config['retro_reac'],
        config['retro_template']
        )
