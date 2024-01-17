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


def load_subsets(retro_reac: str = '', dataset_version: str = '', template_hash_version: str = '', dataset_name: str = '')-> list: 
    '''
    Loads both SMILES and mol subsets composed of molecules from f'{dataset_name}_{dataset_version}' dataset containing SMARTS substructure retro_reac 
    
    --Inputs--
    retro_reac (str):             SMARTS pattern of the substructure to match
    dataset_version (str):        version of the dataset (str) being any integer from 1 to 100.
    template_hash_version (str):  Allows to trace back the template to the templates dataframe, it is constructed as follows. 
                                    template_hash_version = f"{template_hash}_{template_line}"
    dataset_name (str):           name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 100 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 100 must be present in the folder dataset_balance/data/

    --Returns--
    dataset_sub (list):           list of smiles strings containing the substructure matches. Subset of the input dataset.
    dataset_sub_mol (list):       list of mol objects corresponding to the smiles strings containing the substructure matches. Subset of the input dataset_mol.
    '''
    # Define paths and file names
    folder_path     = f'./results/datasets/{dataset_name}'
    folder_path_mol = f'./results/datasets/{dataset_name}_mol'
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

    # Load the subsets if they exist
    try:
        with open(f'{folder_path}/{name}.txt', 'r') as f:
            dataset_sub = [line.strip() for line in f]
        with open(f'{folder_path_mol}/{name}.pkl', 'rb') as f:
            dataset_sub_mol = pickle.load(f)
        return dataset_sub, dataset_sub_mol

    # Print error message if the subsets do not exist
    except FileNotFoundError:
        print(f'No subsets found for retro_reac: {retro_reac} in dataset version: {dataset_version}')
        return [], []
    
    except EOFError:
        print(f'Pickle load ran out of input at retro_reac: {retro_reac} in dataset version: {dataset_version}')
        return [], []

def apply_rxn_template_on_mols_list(dataset_mol:list, rxn_template:str) -> list:
    '''
    Applies a reaction template (rxn_template) on a list of Chem.Mol products (dataset_mol) and returns a list of lists of the corresponding reactants 

    --Inputs--
    dataset_mol (list(Chem.Mol)):   list of molecules mols 
    rxn_template (str):             reaction template in SMART format
    
    --Returns--
    list of lists of reactants
    '''
    # Convert str template to reaction object
    rxn = AllChem.ReactionFromSmarts(rxn_template)

    # Apply reaction object on the product molecules from dataset_mol to obtain the reactants
    return [rxn.RunReactants((dataset_mol[i],)) for i in range(len(dataset_mol))]


def canonicalize(smiles: str) -> str:
    '''
    Converts a smile string into a rdkit canonicalized smile string

    --Inputs--
    smiles (str):   smile string to canonicalize

    --Returns--
    (str) canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def format_reaction(reactants_tuple: tuple, smi : str) -> list:
    '''Formats canonical reactions as reactants>>product in a smiles format where reactants come from reactants_tuple and the product is smi.

    --Inputs--
    reactants_tuple (tuple(Chem.Mol)):    tuple of reactants Chem.Mol objects resulting from applying a reaction template on a molecule Chem.Mol object
    smi (str):                            product smiles string

    --Returns--
    (list) list of canonicalized reactions in a smiles format
    '''
    # Create a list that will contain the different possible reactions for the given product
    reactants_smiles_list = []

    # Iterate over the different possible reactants combinations for the given product
    for i in range(len(reactants_tuple)):
        reactants_mol = list(reactants_tuple[i])
        reactants_smiles = ''

        # Iterate over the several reactants for a single template application
        for j in range(len(reactants_mol)):

            # Append the reactants in a converted smiles format to the reactants_smiles string
            reactants_smiles += Chem.MolToSmiles(reactants_mol[j]) + '.'

        # Remove the last dot and append to the list containing all the different possible reactions
        reactants_smiles = reactants_smiles[:-1]
        reactants_smiles_list.append(reactants_smiles)
    
    # Remove duplicates
    reactants_smiles_list = list(set(reactants_smiles_list))

    # Canonicalize the reactants smiles
    rxn = [canonicalize(reactants_smiles_list[i]) + '>>' + smi for i in range(len(reactants_smiles_list))]
    
    return rxn


def save_rxns(rxns_list, retro_reac, retro_template, dataset_version: str = '', template_hash_version: str = '', dataset_name: str = ''):
    '''
    Saves the rxn list in a txt file, retro_reac is the SMARTS pattern of the product of the reaction,
    retro_template is the template that is applied on retro_reac
    '''

    #Remove the slash from the template
    retro_template = retro_template.replace('/', 'slash')

    if rxns_list:
        folder_path     = f'./results/created_rxns/{dataset_name}'
        name           = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

        temp_path       = f'./results/temp_files/{dataset_name}_temp'
        temp_name       = f'{dataset_name}_temp_{template_hash_version}'
        temp_list       = []

        #Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in rxns_list:
                f.write(item + '\n')

            temp_list.append(f'{folder_path}/{name}.txt')

        #Save the paths of saved subsets to a temp file to delete them once they are no longer needed
        # 1. Create the folder if it does not exist
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
               
        # 2. Create the temp file for the given dataset_name, and template_hash_version
        if not os.path.exists(f'{temp_path}/{temp_name}.txt'):
            with open(f'{temp_path}/{temp_name}.txt', 'w') as f:
                for item in temp_list:
                    f.write(item + '\n')
        else:
            with open(f'{temp_path}/{temp_name}.txt', 'a') as f:
                for item in temp_list:
                    f.write(item + '\n')
    
        print(f'Created {len(rxns_list)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')


def process_retro_template(retro_reac, retro_template, dataset_version: str = '', template_hash_version: str = '', dataset_name: str = ''):
    
    dataset_sub, dataset_sub_mol = load_subsets(retro_reac, dataset_version, template_hash_version, dataset_name)
    
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
    try:
        fictive_rxns_list.remove('>>') #remove empty reactions
    except ValueError:
        pass

    # Save in a txt file
    save_rxns(fictive_rxns_list, retro_reac, retro_template, dataset_version, template_hash_version, dataset_name)


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(dataset_name, dataset_version, template_hash_version, retro_reac, retro_template):
    
    process_retro_template(retro_reac, retro_template, dataset_version, template_hash_version, dataset_name)


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
        #config['template_version'],
        config['template_hash_version'], #new -------------
        config['retro_reac'],
        config['retro_template']
        )
