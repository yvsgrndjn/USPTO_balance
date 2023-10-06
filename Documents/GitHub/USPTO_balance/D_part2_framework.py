# Part 2 - gpt proposal for cleaner version
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

def load_subsets(retro_reac, GDB_version: str = '', template_version: str = ''): 

    name = f'GDB13S_sub_{retro_reac}'
    folder_path = f'./GDB_subsets_{GDB_version}_{template_version}'
    folder_path_mol = f'./GDB_subsets_mol_{GDB_version}_{template_version}'

    try:
        with open(f'{folder_path}/{name}.txt', 'r') as f:
            GDB13S_sub = [line.strip() for line in f]
        with open(f'{folder_path_mol}/{name}.pkl', 'rb') as f:
            GDB13S_sub_mol = pickle.load(f)

        return GDB13S_sub, GDB13S_sub_mol

    except FileNotFoundError:
        print(f'No subsets found for retro_reac: {retro_reac}')
        return [], []


def save_rxns(rxns_list, retro_reac, retro_template, GDB_version: str = '', template_version: str = ''):
    if rxns_list:
        name = f'rxns_{retro_reac}_{retro_template}'
        folder_path = f'./created_rxns_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in rxns_list:
                f.write(item + '\n')
        print(f'Saved {len(rxns_list)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template} and folder : {GDB_version}_{template_version}')


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

def canonicalize(smiles):
    return singlestepretrosynthesis.canonicalize_smiles(smiles)

def format_reaction(reactants_tuple, smi):
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


def process_retro_template(retro_reac, retro_template, GDB_version: str = '', template_version: str = ''):
    
    GDB13S_sub, GDB13S_sub_mol = load_subsets(retro_reac, GDB_version, template_version)

    if not GDB13S_sub:
        return

    # Apply template
    GDB13S_sub_app_temp = apply_rxn_template_on_mols_list(GDB13S_sub_mol, retro_template)

    # Find indices of empty elements and remove them from both lists
    ind_remove = [GDB13S_sub_app_temp[i] == () for i in range(len(GDB13S_sub_app_temp))]
    GDB13S_sub_app_temp_sort = [GDB13S_sub_app_temp[i] for i in range(len(GDB13S_sub_app_temp)) if not ind_remove[i]]
    GDB13S_sub_sort = [GDB13S_sub[i] for i in range(len(GDB13S_sub)) if not ind_remove[i]]

    # Create fictive reactions
    fictive_rxns_list = [format_reaction(GDB13S_sub_app_temp_sort[k], GDB13S_sub_sort[k]) for k in range(len(GDB13S_sub_sort))]
    fictive_rxns_list = list(chain.from_iterable(fictive_rxns_list))

    # Save in a txt file
    save_rxns(fictive_rxns_list, retro_reac, retro_template, GDB_version, template_version)


def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(df_templates_path, GDB_version, template_version):
    
    #load df_templates prepared in (*)
    df_templates = pd.read_pickle(df_templates_path)

    for retro_reac, retro_template in tqdm(zip(df_templates['retro_reac'], df_templates['retro_templates'])):
        process_retro_template(retro_reac, retro_template, GDB_version, template_version)

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
        config['df_templates_path_to_pkl'],
        config['GDB_version'],
        config['template_version']
    )