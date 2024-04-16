import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import pickle
from itertools import chain
from itertools import islice
import yaml
from tqdm import tqdm
import math
import random
from multiprocessing import Pool
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()

from uspto_balance.C_part1_framework import main as c_part1_framework
from uspto_balance.D_part2_framework import main as d_part2_framework
from uspto_balance.E_part3_framework import main as e_part3_framework

#Module containing all functions related to the balancing dataset workflow

# Utilities functions ------------------------------


def canonicalize(smiles: str) -> str:
    '''
    Converts a smile string into a rdkit canonicalized smile string

    --Inputs--
    smiles (str):   smile string to canonicalize

    --Returns--
    (str) canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


#C_part1_framework specific functions ------------------------------


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


#D_part2_framework functions ------------------------------


def load_subsets(retro_reac: str = '', dataset_version: str = '', template_hash_version: str = '', dataset_name: str = '')-> list: 
    '''
    Loads both SMILES and mol subsets composed of molecules from f'{dataset_name}_{dataset_version}' dataset containing SMARTS substructure retro_reac 
    
    --Inputs--
    retro_reac (str):             SMARTS pattern of the substructure to match
    dataset_version (str):        version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):  Allows to trace back the template to the templates dataframe, it is constructed as follows. 
                                    template_hash_version = f"{template_hash}_{template_line}"
    dataset_name (str):           name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 1000 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 1000 must be present in the folder dataset_balance/data/

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


def save_rxns(rxns_list, retro_reac, retro_template, dataset_version: str = '', template_hash_version: str = '', dataset_name: str = ''):
    '''
    Saves the rxn list in a txt file, retro_reac is the SMARTS pattern of the product of the reaction,
    retro_template is the template that is applied on retro_reac

    --Inputs--
    rxns_list (list):               list of reactions in a smiles format
    retro_reac (str):               SMARTS pattern of the substructure to match
    retro_template (str):           reaction template in SMART format
    dataset_version (str):          version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows.
                                    template_hash_version = f"{template_hash}_{template_line}"
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 1000 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 1000 must be present in the folder dataset_balance/data/

    --Returns--
    None, but creates a txt file containing the rxns_list
    '''

    #Remove the slash from the template
    retro_template = retro_template.replace('/', 'slash')

    if rxns_list:
        folder_path     = f'./results/created_rxns/{dataset_name}'
        name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

        temp_path       = f'./results/temp_files/{dataset_name}_temp'
        temp_name       = f'{dataset_name}_temp_{template_hash_version}'
        temp_list       = []

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in rxns_list:
                f.write(item + '\n')

            temp_list.append(f'{folder_path}/{name}.txt')

        # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
        save_created_files_to_temp_file(temp_path, temp_name, temp_list)
        print(f'Created {len(rxns_list)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')

    else:
        print(f'No reactions created for retro_reac: {retro_reac} and retro_template: {retro_template}')



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


def process_retro_template(retro_reac, retro_template, dataset_version: str = '', template_hash_version: str = '', dataset_name: str = ''):
    '''
    Function creating fictive reactions from a given retrosynthetic reaction template 'retro_template' applied on molecules containing a given substructure 'retro_reac'. 
    It is subdivised in several steps: 
        1) load the subsets of molecules containing the substructure 'retro_reac' from the dataset version 'dataset_version' (created in c_part1_framework.py module)
        2) apply the template 'retro_template' on the molecules from the subsets
        3) format the reactions in a smiles format
        4) save the reactions in a txt file.
    
    --Inputs--
    retro_reac (str):               SMARTS pattern of the substructure to match
    retro_template (str):           reaction template in SMART format
    dataset_version (str):          version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows.
                                    template_hash_version = f"{template_hash}_{template_line}"
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset is divided in 1000 different parts

    --Returns--
    None, but creates a txt file containing the fictive reactions 'rxns_list'
    '''
    # Load the subsets of molecules containing the substructure 'retro_reac' from the dataset version 'dataset_version'
    dataset_sub, dataset_sub_mol = load_subsets(retro_reac, dataset_version, template_hash_version, dataset_name)
    
    # Abort if no subset is found
    if not dataset_sub:
        return

    # Apply reaction template on the product molecules from datasets
    dataset_sub_app_temp = apply_rxn_template_on_mols_list(dataset_sub_mol, retro_template)

    # Find indices of empty elements and remove them from both lists
    ind_remove = [result == () for result in dataset_sub_app_temp]
    dataset_sub_app_temp_sort = [dataset_sub_app_temp[i] for i in range(len(dataset_sub_app_temp)) if not ind_remove[i]]
    dataset_sub_sort = [dataset_sub[i] for i in range(len(dataset_sub)) if not ind_remove[i]]

    # Create fictive reactions as reactants>>product in a smiles format
    fictive_rxns_list = [format_reaction(dataset_sub_app_temp_sort[k], dataset_sub_sort[k]) for k in range(len(dataset_sub_sort))]
    fictive_rxns_list = list(chain.from_iterable(fictive_rxns_list))
    try:
        fictive_rxns_list.remove('>>') # Remove empty reactions
    except ValueError:
        pass

    # Save in a txt file
    save_rxns(fictive_rxns_list, retro_reac, retro_template, dataset_version, template_hash_version, dataset_name)


#E_part3_framework functions ------------------------------


def load_rxns(dataset_name: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template: str, rxns_number: int = 10000):
    '''
    Loads the fictive reactions ('rxns_list') that were created in D_part2_framework.py of the dataset subset matching the retro_reac pattern
    and applying reaction templates 'retro_template' to them.

    --Inputs--
    dataset_name (str):           Name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 100 different
                                    files in the format {dataset_name}_i.txt for i from 1 to 1000 must be present in the folder dataset_balance/data/
    dataset_version (str):        Version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):  Allows to trace back the template to the templates dataframe, it is constructed as follows. 
    retro_reac (str):             SMARTS pattern of the substructure to match
    retro_template (str):         Reaction template in SMART format
    rxns_number:                  Number of reactions to load (default = 10000)
    '''
    try:
        folder_path = f'./results/created_rxns/{dataset_name}'
        name =         f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

        with open(f'{folder_path}/{name}.txt', 'r') as f:
            rxns_list = []
            rxns_list = list(islice(f, rxns_number))
            rxns_list = [rxns_list[i].split('\n')[0] for i in range(len(rxns_list))]
        return rxns_list

    except FileNotFoundError:
        print(f'No reactions found for retro_reac: {retro_reac} and retro_template: {retro_template}')
        return []


def remove_incomplete_rxns(rxns_list:list):
    '''
    Removes reactions that have no reactant (= template application on product did not work out)

    --Inputs--
    rxns_list (list):             List of reactions supposedly in the format reactants>>product but that might contain >>product reactions that will be filtered out

    --Returns--
    rxns_list (list):             List of reactions only in the format reactants>>product
    '''
    indices_to_remove = [i for i, value in enumerate(rxns_list) if value[0] == '>']

    for index in reversed(indices_to_remove):
        del rxns_list[index]
    return rxns_list


def tokenize_rxn_list(rxns_list: list):
    '''
    Tokenizes a list of reactions

    --Inputs--
    rxns_list (list):             List of reactions in the format reactants>>product

    --Returns--
    tok_rxns_list (list):         List of tokenized reactions in the format r e a c t a n t s >> p r o d u c t
    '''
    tok_rxns_list = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_list]

    return tok_rxns_list


def run_T2_predictions(tok_rxns_list: list, Model_path: str, beam_size: int = 1, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>>product) and predicts the needed reagents for the reaction to take place.
    Gives back the list of reagents, which is same length as the input SMILES list.
    --Inputs--
    tok_rxns_list (list(str)):   List of tokenized reactions in the format reactants>>product
    Model_path (str):             Path to the model to use for prediction
    beam_size (int):              Beam size to use for prediction (default = 1)
    batch_size (int):             Batch size to use for prediction (default = 64)
    untokenize_output (bool):     Whether to untokenize the output or not (default = True)

    --Returns--
    preds_T2 (list(str)):         List of predicted reagents for each of the input reactions of 'tok_rxns_list'
    '''
    [preds_T2, probs_T2] = singlestepretrosynthesis.Execute_Prediction(tok_rxns_list, Model_path, beam_size, batch_size, untokenize_output)
    
    return preds_T2[0]


def remove_unmapped_rxns(MappedReactions: list, preds_T2: list, rxns_list: list):
    '''
    Remove the reactions for which mapping was unsuccessful, characterised by '>>'

    --Inputs--
    MappedReactions (list(str)):    List of mapped reactions in the format reactants(mapped)>>product(mapped)
    preds_T2 (list(str)):           List of predicted reagents for each of the input reactions of 'rxns_list'
    rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped)

    --Returns--
    MappedReactions (list(str)):    List of mapped reactions in the format reactants(mapped)>>product(mapped) without the unmapped reactions
    preds_T2 (list(str)):           List of predicted reagents for each of the input reactions of 'rxns_list' without the predictions corresponding to the unmapped reactions
    rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped) without the reactions corresponding to the unmapped reactions
    '''
    indices_to_remove = [i for i, value in enumerate(MappedReactions) if value == '>>']

    for index in reversed(indices_to_remove):
        del MappedReactions[index]
        del preds_T2[index]
        del rxns_list[index]
    return MappedReactions, preds_T2, rxns_list


def prepare_rxns_T2_for_T3(rxns_list: list, preds_T2: list):
    '''
    From the rxns_list and the predicted reagents for each reaction, returns a list of tokenized reactions
    in an appropriate format to use as input to forward tag T3
    reactants tagged "!" > reagents (tokenized)

    --Inputs--
    rxns_list (list(str)):              List of reactions in the format reactants>>product
    preds_T2 (list(str)):               List of predicted reagents for each reaction in 'rxns_list'

    --Returns--
    reconstructed_rxns_tok (list(str)): List of tokenized reactions in the format: reactants(tagged with "!") > reagents (tokenized) ready to be input into forward-tag Forward validation (T3-FT)
    '''
    MappedReactions = list(singlestepretrosynthesis.rxn_mapper_batch.map_reactions(rxns_list))
    MappedReactions, preds_T2, rxns_list = remove_unmapped_rxns(MappedReactions, preds_T2, rxns_list)
    taggedreactants = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = True, tag_reactants = True).split('>>')[0] for i in range(len(MappedReactions))]
    reconstructed_rxns = [taggedreactants[i] + '>' + preds_T2[i] for i in range(len(preds_T2))]
    reconstructed_rxns_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in reconstructed_rxns]
    return reconstructed_rxns_tok


def run_T3_predictions(rxns_T2_to_T3_tok: list, Model_path: str, beam_size: int = 3, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>reagents) and performs forward prediction on them.
    Gives back the list of predicted products that has the same length as the input SMILES list.  

    --Inputs--
    rxns_T2_to_T3_tok (list(str)):   List of tokenized reactions in the format: reactants(tagged with "!") >reagents (tokenized)
    Model_path (str):                Path to the model to use for prediction (here forward validation, preferably with forward-tag)
    beam_size (int):                 Beam size to use for prediction (default = 3)
    batch_size (int):                Batch size to use for prediction (default = 64)
    untokenize_output (bool):        Whether to untokenize the output or not (default = True)

    --Returns--
    preds_T3[0] (list(str)):            List of predicted products for each of the input reactions of 'rxns_T2_to_T3_tok'
    probs_T3[0] (list(float)):          List of confidence scores for each of the predicted products of 'preds_T3', in [0,1]
    '''
    [preds_T3, probs_T3] = singlestepretrosynthesis.Execute_Prediction(rxns_T2_to_T3_tok, Model_path, beam_size, batch_size, untokenize_output)
    return preds_T3[0], probs_T3[0]


def find_ind_match_T3_preds_ref(preds_T3: list, rxns_list: list):
    '''
    Performs forward validation on the predictions of T3 and the original reactions list (used as ground truth). Returns the matches between the two lists.
    Takes as input the forward predictions of T3 and the original reactions list, returns the indices of the predicted products matching the original
    products (returns indices of forward validated reactions)

    --Inputs--
    preds_T3 (list(str)):           List of predicted products for each of the input reactions of 'rxns_T2_to_T3_tok' (output from run_T3_predictions)
    rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped) representing the ground truth

    --Returns--
    ind_match (list(int)):          List of indices of the forward validated reactions
    '''
    # Canonicalization of the predictions (might have been done already in the prediction process)
    preds_T3 = [singlestepretrosynthesis.canonicalize_smiles(i) for i in preds_T3]
    
    # Compare predictions and ground truth (original products), keep indices of matches
    preds_ref = [singlestepretrosynthesis.canonicalize_smiles(rxns_list[i].split('>>')[1]) for i in range(len(rxns_list))] 
    ind_match = [i for i in range(len(preds_T3)) if preds_T3[i] == preds_ref[i]]
    return ind_match


def add_reagents_to_rxns_list(rxns_list, preds_T2, ind_match):
    '''
    Add reagents to the list of reactions 'rxns_list' that are forward validated (at indices 'ind_match') . Final reaction format: reactant(s)>reagent(s)>product.

    --Inputs--
    rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped)
    preds_T2 (list(str)):           List of predicted reagents for each reaction in 'rxns_list'
    ind_match (list(int)):          List of indices of the forward validated reactions

    --Returns--
    rxns_list_with_reagents (list(str)): List of validated reactions with format: reactant(s)>reagent(s)>product
    '''
    rxns_list_with_reagents = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] + '>' + rxns_list[i].split('>>')[1] for i in ind_match]
    return rxns_list_with_reagents


def keeps_val_rxns_and_scores(rxns_list, probs_T3, ind_match): 
    '''
    Function returning the confidence scores of the forward validated reactions and the reactions themselves.

    --Inputs--
    rxns_list (list(str)):          List of forward validated reactions
    probs_T3 (list(float)):         List of confidence scores for each of the predicted products of 'preds_T3', in [0,1]
    ind_match (list(int)):          List of indices of the forward validated reactions

    --Returns--
    rxns_val (list(str)):           List of forward validated reactions (no modification from 'rxns_list')
    conf_scores (list(float)):      List of confidence scores for each of the forward validated reactions, in [0,1]
    '''
    rxns_val = rxns_list
    conf_scores = [probs_T3[i] for i in ind_match]
    return rxns_val, conf_scores


def save_rxns_and_conf_to_csv(rxns_val: list, conf_scores: list, dataset_name: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template: str):
    '''
    Saves the validated reactions and their confidence scores in a csv file.

    --Inputs--
    rxns_val (list(str)):           List of forward validated reactions
    conf_scores (list(float)):      List of confidence scores for each of the forward validated reactions, in [0,1]
    dataset_name (str):             Name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function):
                                    The dataset divided in 1000 different files in the format {dataset_name}_i.txt for i from 1 to
                                    1000 must be present in the folder dataset_balance/data/
    dataset_version (str):          Version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows.
    retro_reac (str):               SMARTS pattern of the substructure to match
    retro_template (str):           Reaction template in SMART format
    '''
    retro_template = retro_template.replace('/', 'slash')

    folder_path     = f'./results/saved_rxns/{dataset_name}'
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

    temp_path       = f'./results/temp_files/{dataset_name}_temp'
    temp_name       = f'{dataset_name}_temp_{template_hash_version}'
    temp_list       = []

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create and save a dataframe with the reactions and the confidence scores
    df = pd.DataFrame({'rxns': rxns_val, 'conf_scores': conf_scores})
    df.to_csv(f'{folder_path}/{name}.csv', index=False)

    temp_list.append(f'{folder_path}/{name}.csv')

    # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
    save_created_files_to_temp_file(temp_path, temp_name, temp_list)
    print(f'Validated and saved {len(rxns_val)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')


def reactions_conf_validation(dataset_name: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template: str, Model_path_T2: str, Model_path_T3: str):
    '''
    Takes the fictive reactions created in D_part2_framework.py, predicts the reagents needed for the reaction to take place, and performs
    forward validation of the reactants>reagents reactions. Saves the validated reactions and their confidence scores in a csv file.

    --Inputs--
    dataset_name (str):             Name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function):
                                    The dataset divided in 1000 different files in the format {dataset_name}_i.txt for i from 1 to 1000
                                    must be present in the folder dataset_balance/data/
    dataset_version (str):          Version of the dataset (str) being any integer from 1 to 1000.
    template_hash_version (str):    Allows to trace back the template to the templates dataframe, it is constructed as follows.
    retro_reac (str):               SMARTS pattern of the substructure to match
    retro_template (str):           Reaction template in SMART format
    Model_path_T2 (str):            Path to the model to use for prediction (here reagent prediction)
    Model_path_T3 (str):            Path to the model to use for prediction (here forward validation, preferably with forward-tag for better results)

    --Returns--
    None, but saves the validated reactions and their confidence scores in a csv file.
    '''
    rxns_list = load_rxns(dataset_name, dataset_version, template_hash_version, retro_reac, retro_template)

    if not rxns_list:
        return
    rxns_list = remove_incomplete_rxns(rxns_list)
    tok_rxns_list = tokenize_rxn_list(rxns_list)
    preds_T2 = run_T2_predictions(tok_rxns_list, Model_path_T2, beam_size = 1, batch_size = 64, untokenize_output = True)
    rxns_T2_to_T3_tok = prepare_rxns_T2_for_T3(rxns_list, preds_T2)
    preds_T3, probs_T3 = run_T3_predictions(rxns_T2_to_T3_tok, Model_path_T3, beam_size = 3, batch_size = 64, untokenize_output = True)
    ind_match = find_ind_match_T3_preds_ref(preds_T3, rxns_list)
    rxns_list_with_reagents = add_reagents_to_rxns_list(rxns_list, preds_T2, ind_match)
    rxns_val, conf_scores = keeps_val_rxns_and_scores(rxns_list_with_reagents, probs_T3, ind_match)
    save_rxns_and_conf_to_csv(rxns_val, conf_scores, dataset_name, dataset_version, template_hash_version, retro_reac, retro_template)


#dataset_equilibration functions ------------------------------


def extract_subset_from_dataset(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder):
    '''
    Performs the first part of the dataset equilibration: extracts a subset of the dataset to be used for the enrichment. For a given retrosynthetic
    reaction template 'retro_template' and the typical substructure on the product-side of the reaction 'retro_reac', the function extracts all the molecules
    the 'retro_reac' substructure from the dataset (the dataset is a subset of the complete dataset_name : {dataset_name}_{dataset_version}). The results are
    a list of molecules and their corresponding Chem.Mol objects, and they will be saved for further use by parts 2 and 3.

    --Inputs--
    dataset_name (str):         name of the dataset to be used for the enrichment
    dataset_version (int):      version of the dataset to be used for the enrichment
    retro_reac (str):           typical substructure on the product-side of the reaction
    retro_template (str):       retrosynthetic reaction template
    template_hash (str):        hash of the retrosynthetic reaction template
    template_line (int):        line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    path_to_folder (str):       path to the folder containing the config_files and the data folder

    --Returns--
    None, but creates a txt file containing molecules matching the 'retro_reac' substructure from the dataset, and a pkl file containing the corresponding
    Chem.Mol objects. These files will be used by parts 2 and 3.
    '''
    print('In extract_subset_from_dataset')

    #initialize config variables
    template_hash_version      = f"{template_hash}_{template_line}"
    dataset_path               = f"{path_to_folder}data/{dataset_name}_{dataset_version}.txt"

    # Check if part 1 needs to be run
    folder_path = f'./results/datasets/{dataset_name}'
    name        = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'

    temp_path       = f'./results/temp_files/{dataset_name}_temp'
    temp_name       = f'{dataset_name}_temp_{template_hash_version}'
    temp_list       = []

    # Run part 1 if the subset does not exist yet
    if not os.path.exists(f'{folder_path}/{name}.txt'):

        # Replace the ';' characters in the config file name
        config_file_path = f'./config_files/config_part1_{dataset_name}_{template_hash_version}.yaml'.replace(';','')

        # Create config file for part 1
        with open(f'{config_file_path}', 'w') as f:

            f.write(f'dataset_name: \'{dataset_name}\'\n')
            f.write(f'dataset_path: \'{dataset_path}\'\n')
            f.write(f'dataset_version: \'{dataset_version}\'\n')
            f.write(f'template_hash_version: \'{template_hash_version}\'\n')
            f.write(f'retro_reac: \'{retro_reac}\'\n')
            f.write(f'retro_template: \'{retro_template}\'\n')

        temp_list.append(f'{config_file_path}')
        
        # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
        save_created_files_to_temp_file(temp_path, temp_name, temp_list)

        # Run part 1
        with open(f'{config_file_path}', 'r') as f:
            config1 = yaml.safe_load(f)

        c_part1_framework(          
            config1['dataset_name'],
            config1['dataset_path'],
            config1['dataset_version'],
            config1['template_hash_version'],
            config1['retro_reac'],
            config1['retro_template']
            )


def create_reactions_using_template(dataset_name: str, dataset_version: str, retro_reac: str, retro_template: str, template_hash: str, template_line: str, path_to_folder: str):
    '''
    Creates fictive reactions from a given retrosynthetic reaction template 'retro_template' applied on molecules containing a given substructure 'retro_reac'. 
    It is subdivised in several steps: 
        1) load the subsets of molecules containing the substructure 'retro_reac' from the dataset version 'dataset_version' (created in c_part1_framework.py module)
        2) apply the template 'retro_template' on the molecules from the subsets
        3) format the reactions in a smiles format
        4) save the reactions in a txt file.
    
    --Inputs--
    dataset_name (str):         name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 1000 different
                                versions, each version being a subset of the complete dataset. The dataset is named {dataset_name}_{dataset_version} with dataset_version
    dataset_version (str):      version of the dataset (str) being any integer from 1 to 1000.
    retro_reac (str):           typical substructure on the product-side of the reaction
    retro_template (str):       retrosynthetic reaction template
    template_hash (str):        hash of the retrosynthetic reaction template
    template_line (int):        line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    path_to_folder (str):       path to the folder containing the config_files and the data folder

    --Returns--
    None, but creates a txt file containing the reactions created from the template 'retro_template' applied on molecules containing the substructure 'retro_reac'.
    '''
    print('In create_reactions_using_template')

    template_hash_version      = f"{template_hash}_{template_line}"

    temp_path       = f'./results/temp_files/{dataset_name}_temp'
    temp_name       = f'{dataset_name}_temp_{template_hash_version}'
    temp_list       = []

    config_file_path2 = f'{path_to_folder}config_files/config_part2_{dataset_name}_{template_hash_version}.yaml'.replace(';','')

    # Create config file for part 2
    with open(f'{config_file_path2}', 'w') as f:
        f.write(f'dataset_name: \'{dataset_name}\'\n')
        f.write(f'dataset_version: \'{dataset_version}\'\n')
        f.write(f'template_hash_version: \'{template_hash_version}\'\n')
        f.write(f'retro_reac: \'{retro_reac}\'\n')
        f.write(f'retro_template: \'{retro_template}\'\n')

    temp_list.append(f'{config_file_path2}')

    # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
    save_created_files_to_temp_file(temp_path, temp_name, temp_list)

    # run part 2
    with open(f'{config_file_path2}', 'r') as f:
        config2 = yaml.safe_load(f)
    
    d_part2_framework(
        config2['dataset_name'],
        config2['dataset_version'],
        config2['template_hash_version'],
        config2['retro_reac'],
        config2['retro_template']
    )


def validate_created_reactions(dataset_name: str, dataset_version: str, retro_reac: str, retro_template: str, template_hash: str, template_line: str, path_to_folder: str, path_models):
    '''
    Takes the fictive reactions created in D_part2_framework.py, predicts the reagents needed for the reaction to take place, and performs
    forward validation of the reactants>reagents reactions. Saves the validated reactions and their confidence scores in a csv file.

    --Inputs--
    dataset_name (str):         name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset divided in 1000 different
                                versions, each version being a subset of the complete dataset. The dataset is named {dataset_name}_{dataset_version} with dataset_version
    dataset_version (str):      version of the dataset (str) being any integer from 1 to 1000.
    retro_reac (str):           typical substructure on the product-side of the reaction
    retro_template (str):       retrosynthetic reaction template
    template_hash (str):        hash of the retrosynthetic reaction template
    template_line (int):        line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    path_to_folder (str):       path to the folder containing the config_files and the data folder

    --Returns--
    None, but creates a csv file containing the validated reactions and their confidence scores.
    '''
    print('In validate_created_reactions')
    
    template_hash_version      = f"{template_hash}_{template_line}"
    Model_path_T2              = f"{path_models}USPTO_STEREO_separated_T2_Reagent_Pred_225000.pt"
    Model_path_T3              = f"{path_models}T3_Fwd_Tag_model_step_300000.pt"
    
    temp_path       = f'./results/temp_files/{dataset_name}_temp'
    temp_name       = f'{dataset_name}_temp_{template_hash_version}'
    temp_list       = []

    config_file_path3 = f'{path_to_folder}config_files/config_part3_{dataset_name}_{template_hash_version}.yaml'.replace(';','')

    # Create config file for part 3
    with open(f'{config_file_path3}', 'w') as f:
        f.write(f'dataset_name: \'{dataset_name}\'\n')
        f.write(f'dataset_version: \'{dataset_version}\'\n')
        f.write(f'template_hash_version: \'{template_hash_version}\'\n')
        f.write(f'retro_reac: \'{retro_reac}\'\n')
        f.write(f'retro_template: \'{retro_template}\'\n')
        f.write(f'Model_path_T2: {Model_path_T2}\n')
        f.write(f'Model_path_T3: {Model_path_T3}')

    temp_list.append(f'{config_file_path3}')

    # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
    save_created_files_to_temp_file(temp_path, temp_name, temp_list)

    # run part 3
    with open(f'{config_file_path3}', 'r') as f:
        config3 = yaml.safe_load(f)
    
    e_part3_framework(
        config3['dataset_name'],
        config3['dataset_version'],
        config3['template_hash_version'],
        config3['retro_reac'],
        config3['retro_template'],
        config3['Model_path_T2'],
        config3['Model_path_T3']
    )


def get_txt_file(path_to_file: str):
    '''
    Loads a txt file into a list

    --Inputs--
    path_to_file (str): path to the txt file to be loaded

    --Returns--
    file (list): list containing the lines of the txt file
    '''
    with open(path_to_file, 'r') as f:
        file = []
        for line in f:
            file.append(line.split('\n')[0])
    return file


def append_df_to_csv(path_to_csv: str, df_to_append: pd.DataFrame):
    '''
    Appends a dataframe to a csv file

    --Inputs--
    path_to_csv (str):           path to the csv file to be appended
    df_to_append (pd.DataFrame): dataframe to be appended to the csv file

    --Returns--
    None, appends the dataframe to the csv file
    '''
    df_to_append.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index = False)  


def append_n_elements_to_file(path_to_file: str, list_to_append: list, n_elements: int):
    '''
    Appends the first n_elements of a list to a txt file stored under 'path_to_file'

    --Inputs--
    path_to_file (str):     path to the txt file to be appended
    list_to_append (list):  list containing the elements to be appended to the txt file
    n_elements (int):       number of elements to be appended to the txt file

    --Returns--
    None, appends the first n_elements of the list to the txt file
    '''
    with open(path_to_file, 'a') as f:
        for i in range(n_elements):
            f.write(list_to_append[i] + '\n')


def delete_all_files_from_list(list_of_paths: list):
    '''
    Takes a list of paths as input and deletes all the files and folders contained in the list of paths

    --Inputs--
    list_of_paths (list): list of paths to be deleted

    --Returns--
    None, deletes all the files and folders contained in the list of paths
    '''
    for file_path in list_of_paths:
        if 'full' in file_path:
            continue
        else:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def append_saved_rxns_until_enrichment_target_csv(dataset_name: str, dataset_version: str, retro_reac: str, retro_template: str, template_hash: str, template_line: str, template_frequency: int, frequency_target: int = 10000):
    '''
    Appends all validated reactions for a given dataset_name, dataset_version, retro_reac, retro_template, template_hash, and template_line to the other validated reactions for the same parameters but dataset_version until the number
    of validated reactions with confident reactions reaches the frequency_target. The validated reactions are all saved under a csv file. 

    --Inputs--
    dataset_name (str):           name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset is divided in 1000 different parts
    dataset_version(str(int)):    version of the dataset (str) being any integer from 1 to 1000.
    retro_reac (str):             SMARTS pattern of the substructure to match
    retro_template (str):         retrosynthetic reaction template in SMART format
    template_hash (str):          hash of the retrosynthetic reaction template
    template_line (str(int)):     line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    template_frequency (int):     Actual number of reactions that for which the extracted template corresponds to 'retro_template'
    frequency_target (int):       Enrichment target for the number of reactions that will show a reactivity corresponding to 'retro_template'

    --Returns--
    None, the validated reactions are saved under a csv file common for a same dataset_name, retro_reac, retro_template, template_hash, and template_line but different dataset_version's
    '''
    print('In append_saved_rxns_until_enrichment_target')

    retro_reac             = retro_reac.replace('/', 'slash')
    retro_template         = retro_template.replace('/', 'slash')
    folder_path            = f'./results/saved_rxns/{dataset_name}'
    template_hash_version  = f"{template_hash}_{template_line}"
    name                   = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'
    name_to_save           = f'{dataset_name}_{template_hash_version}'

    temp_path              = f'./results/temp_files/{dataset_name}_temp'
    temp_name              = f'{dataset_name}_temp_{template_hash_version}'

    # Count the number of created validated and confident enough reactions, append reactions to the csv file of results
    if os.path.exists(f'{folder_path}/{name}.csv'):
        df_saved_rxns = pd.read_csv(f'{folder_path}/{name}.csv')
        numel_conf = sum(i > 0.95 for i in df_saved_rxns['conf_scores'].values.tolist())
        template_frequency += numel_conf
        num_added_rxns = len(df_saved_rxns)
        
        append_df_to_csv(f'{folder_path}/full_{name_to_save}.csv', df_saved_rxns)

    else:
        print(f'No reactions found under: {folder_path}/{name}.csv (append_saved_rxns_until_enrichment_target)')
        num_added_rxns = 0
    
    # Delete all temporary files
    if os.path.exists(f'{temp_path}/{temp_name}.txt'):
        list_of_paths = get_txt_file(f'{temp_path}/{temp_name}.txt')
        try:
            delete_all_files_from_list(list_of_paths)
            os.remove(f'{temp_path}/{temp_name}.txt')
        except:
            print('Could not delete all files from list')
    return template_frequency, num_added_rxns


def add_dataset_fraction_to_csv(dataset_name: str, dataset_version: str, template_hash: str, template_line: str, retro_reac: str, retro_template: str):
    '''
    Appends the number of enrichment runs as the last line of the csv file containing the results of the enrichment under the columns 'rxns'. 

    --Inputs--
    dataset_name (str):           name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset is divided in 1000 different parts
    dataset_version(str(int)):    version of the dataset (str) being any integer from 1 to 1000.
    template_hash (str):          hash of the retrosynthetic reaction template
    template_line (str(int)):     line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    retro_reac (str):             SMARTS pattern of the substructure to match
    retro_template (str):         retrosynthetic reaction template in SMART format
    
    --Returns--
    None, appends number of enrichment runs as the last line of the csv file containing the results of the enrichment under the column 'rxns
    '''
    retro_reac                 = retro_reac.replace('/', 'slash')
    retro_template             = retro_template.replace('/', 'slash')
    folder_path                = f'./results/saved_rxns/{dataset_name}'
    template_hash_version      = f"{template_hash}_{template_line}"
    name_to_save               = f'{dataset_name}_{template_hash_version}'

    path_to_csv = f'{folder_path}/full_{name_to_save}.csv'

    counter = dataset_version + 1
    data = [[counter, None]]
    df_to_append = pd.DataFrame(data, columns = ['rxns', 'conf_scores'])

    df_to_append.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index = False)


#Functions used in other steps than the actual dataset equilibration ------------------------------


def select_templates_to_enrich(data: pd.Series, min_number: int = 5, target_number: int = 100000): 
    """
    Takes a Pandas Series of template hashes and returns the list of hashes that meet enrichment requirements along with their frequencies.

    data:           Pandas Series of template hashes
    min_number:     minimum number of occurrences of a template hash to be considered for enrichment
    target_number:  target number of occurrences of a template hash after enrichment
    """
    # Calculate the number of occurrences for each template hash
    template_counts = data.value_counts()

    # Filter the template counts to select hashes within the specified range
    selected_hashes = template_counts[
        (template_counts >= min_number) & (template_counts <= target_number)
    ]

    # Extract the index (template hashes) as a list
    selected_hashes_list = selected_hashes.index.tolist()

    selected_frequencies_list = selected_hashes.sort_values(ascending=False).tolist()

    return selected_hashes_list, selected_frequencies_list



def rxn_smarts_to_sanitized_reactant_smarts(smarts):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns its reactant side in a rdkit mol format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        smarts = smarts.split('>>')[0]
        if smarts[0]=='(' and smarts[-1] == ')':
            smarts = smarts[1:-1].replace(').(', '.')

        #mol = Chem.MolFromSmarts(smarts)
        #return mol
        return smarts
    except:
        return ''



def rxn_smarts_to_sanitized_product_smarts(smarts):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns its product side in a rdkit mol format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        smarts = smarts.split('>>')[1]
        if smarts[0]=='(' and smarts[-1] == ')':
            smarts = smarts[1:-1].replace(').(', '.')

        #mol = Chem.MolFromSmarts(smarts)
        #return mol
        return smarts
    except:
        return ''



def rxn_smarts_to_formatted_rxn_smarts(smarts:str):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns it in a rdkit acceptable format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        reac = smarts.split('>>')[0]
        prod = smarts.split('>>')[1]
        smarts = smarts.split('>>')[1]
        if smarts[0]=='(' and smarts[-1] == ')':
            reac = reac[1:-1].replace(').(', '.')
            prod = prod[1:-1].replace(').(', '.')
            
        rxnsmarts = AllChem.ReactionFromSmarts(reac + '>>' + prod)
        return rxnsmarts
    except:
        return ''


def find_reaction_template_of_hash(fulltemplate_df:pd.DataFrame, template):
    '''
    For a given template, returns its associated most frequent reaction template (column 'retro_template') from the full template dataframe
    '''
    return fulltemplate_df[fulltemplate_df['template_hash'] == template]['retro_template'].value_counts().keys().tolist()[0]


def load_stats_csv_into_df(path_to_folder: str, dataset_name: str, df_templates: pd.DataFrame):
    '''
    Takes the overall statistics of the enrichment process (dataset_equilibration performed on several templates) and loads them into a dataframe summarizing the results for each template hash.

    --Inputs--
    path_to_folder (str):           path to the folder containing the csv files with the statistics of the enrichment process, typically: f'./results/saved_rxns/{dataset_name}/'
    dataset_name (str):             name of the dataset (str) ex: GDB13S, USPTO
    df_templates (pd.DataFrame):    dataframe containing the information on the templates to be enriched 
    '''
    df_template_stats = pd.DataFrame(columns=['template line', 'dataset fractions', 'molecules match', 'created_rxns', 'validated reactions', 'validated and confident reactions', 'time elapsed', 'cpu type'])
    df_template_stats.set_index('template line', inplace=True)

    for template_line in tqdm(range(len(df_templates))):
        try:
            template_hash = df_templates.at[template_line,'template_hash']
            name_saved = f'full_{dataset_name}_stats_{template_hash}_{template_line}.csv'
            dftemp = pd.read_csv(f'{path_to_folder}{name_saved}')
            dftemp.set_index('template line', inplace=True)
            df_template_stats = pd.concat([df_template_stats, dftemp], axis = 0)
        except FileNotFoundError: 
            # Adds empty lines for the templates that did not have any results
            dftemp = pd.DataFrame({'template line': [template_line]})
            dftemp.set_index('template line', inplace=True)
            df_template_stats = pd.concat([df_template_stats, dftemp], axis = 0)
    return df_template_stats


def find_nan_stats_csv(df_template_stats: pd.DataFrame):
    '''
    Find the NaN rows in the df_template_stats dataframe, which also corresponds to the templates for which no stats.csv file was found. Apply on dataframes created by load_csv_into_df().
    '''
    # bool list
    # 1. find NaN rows (= with no existing stats.csv files)
    temp1 = [ (math.isnan(el)) for el in df_template_stats['dataset fractions'] ]

    # 2. Only look at the lines that are supposed to have a stats.csv file (between 4430 and 10'000)
    temp2 = (df_template_stats.index > 4429) & (df_template_stats.index < 10000)

    # -- combine all the conditions together
    temp = (temp1 & temp2)
    nan_ind_list = df_template_stats[temp].index

    return nan_ind_list


# Functions to be used in the post-processing of the dataset equilibration, to recover the results and prepare datasets ------------------------------

def stats_preprocessing(df: pd.DataFrame):
    """
    Remove all the data that we won't use for the statistical analysis, return a bool list with True values at the indices to be kept for further analysis
    """
    nan_values = [ (math.isnan(el)) for el in df['dataset fractions'] ]
    timelimit = (df['dataset fractions']< 1000) & (df['validated and confident reactions'] < 5000)
    no_matches = (df['molecules match'] == 0)
    no_val_rxn = (df['validated reactions'] == 0)
    no_val_conf_rxn = (df['validated and confident reactions'] == 0)
    temp = [nan_values[i] or no_matches[i] or timelimit[i] or no_val_conf_rxn[i] for i in range(len(nan_values))]
    to_keep = [not el for el in temp]
    
    return to_keep, nan_values, timelimit, no_matches, no_val_rxn, no_val_conf_rxn



def calculate_stats_enrichment(fraction_length: int, df_data_fractions:pd.Series, df_mol_match: pd.Series, df_created_rxn: pd.Series, df_val_rxn: pd.Series, df_val_conf_rxn: pd.Series):
    '''
    Calculates the statistics of the enrichment for each template, needs to be run on previously preprocessed data, removing NaN values, rows with no matches, zero validated reactions, confident reactions, or validated and confident reactions
    '''
    molecules_per_match   = (df_data_fractions * fraction_length)/df_mol_match
    match_per_created_rxn = df_mol_match/df_created_rxn
    created_rxn_per_val_rxn = df_created_rxn/df_val_rxn
    val_rxn_per_val_conf_rxn = df_val_rxn/df_val_conf_rxn

    #overall (number of molecules needed to wander through to obtain one validated and confident reaction)
    molecules_per_val_conf_rxn = (df_data_fractions * fraction_length)/df_val_conf_rxn
    
    return molecules_per_match, match_per_created_rxn, created_rxn_per_val_rxn, val_rxn_per_val_conf_rxn, molecules_per_val_conf_rxn


def append_saved_rxns_into_csv(path_to_folder: str, dataset_name: str, df_templates: pd.DataFrame, csv_name: str = ''):
    '''
    Append all saved_rxns csv files generated by the dataset_equilibration into a same csv file stored under the /results/datasets/{dataset_name}/ folder

    --Inputs--
    path_to_folder(str):         path to the folder in which the calculations were run, must contain ./results/saved_rxns/{dataset_name}/
    dataset_name(str):           name of the dataset used as reactions pool
    df_templates(pd.DataFrame):  dataframe containing the templates used to generate the reactions
    csv_name(str):               name of the csv file to save the results into. If not specified, the default name is 'fictive_{dataset_name}_rxns.csv'
    '''

    index_not_found = []
    append_counter = 0
    for i in tqdm(range(len(df_templates))):
        template_hash = df_templates.at[i,'template_hash']
        template_line = i
        retro_template = df_templates.at[i,'retro_templates']
        try:
            df = pd.read_csv(f'{path_to_folder}/results/saved_rxns/{dataset_name}/full_{dataset_name}_{template_hash}_{template_line}.csv')
            df_to_append = pd.DataFrame(columns=['template_line', 'rxns', 'mapped_rxns', 'conf_scores'])
            df_to_append['rxns'] = df['rxns']
            df_to_append['mapped_rxns'] = df['mapped_rxns']
            df_to_append['conf_scores'] = df['conf_scores']
            df_to_append['template_line'] = template_line
            df_to_append['retro_template'] = retro_template

            # check in case the last row 'rxns' is a number -> remove it (only needed for old saved_rxns files)
            try:
                type(int(df_to_append['rxns'].iloc[-1])) == int
                df_to_append.drop(df_to_append.tail(1).index,inplace=True)
            except:
                pass
            
            # check if the file already exists, if it does, do something (rewrite it from the beginning?)

            # append dataframe_i to the dataframe containing all the saved_rxns results
            if append_counter == 0:
                df_to_append.to_csv(f'{path_to_folder}/results/datasets/{dataset_name}/fictive_{dataset_name}_rxns_{csv_name}.csv')
            else:
                df_to_append.to_csv(f'{path_to_folder}/results/datasets/{dataset_name}/fictive_{dataset_name}_rxns_{csv_name}.csv', mode='a', header=False)
            append_counter += 1
        except:
            index_not_found.append(i)


def add_products_column(df: pd.DataFrame):
    '''
    Add a 'products' column to the dataframe containing all the saved_rxns results by deriving it from the 'rxns' column.

    --Inputs--
    df (pd.DataFrame): The dataframe containing the saved_rxns results. The column 'rxns' must exist and contain the reactions in the SMILES form 'reactants>reagents>products'
    --Outputs--
    df (pd.Dataframe): The dataframe with the added 'products' column

    '''
    if 'rxns' in df.columns:

        df['products'] = [ df.at[i, 'rxns'].split('>')[2] for i in range(len(df)) ]
        return df

    else:

        raise KeyError('The dataframe does not contain a "rxns" column')


def label_reactions_from_testset(df: pd.DataFrame, testset_rxns: pd.Series, reaction_col: str):
    '''
    Attributes to the TEST split reactions present in the 'testset_rxns' Series (that are in the test set of the original dataset).
    Also attributes to the TEST split reactions that have the same products as the test set reactions.

    --Inputs--
    df: pd.DataFrame, the dataframe containing the reactions to be split
    testset_rxns: pd.Series, the series containing the reactions that are in the test set of the original dataset
    reaction_col: str, the name of the column containing the reactions to be reformatted in df
    --Outputs--
    df: pd.DataFrame, the original dataframe containing the new 'Set' column, initialized to '' and attributed 'TEST' to the reactions in the test set from uspto (1) and the ones sharing the same products with reactions from (1)
    '''
    # format the reactions from A>B>C to A>>C
    df['Reaction_A>>C'] = df[reaction_col].apply(lambda x: x.split('>')[0] + '>>' + x.split('>')[2])

    # Determine which reactions are in the test set from uspto
    df['isin_uspto_testset'] = df['Reaction_A>>C'].isin(testset_rxns)

    # Initialize all sets to '' and attribute 'TEST' to the reactions in uspto test set
    df['Set'] = ''
    df['Set'].loc[df['isin_uspto_testset'] == True] = 'TEST'

    # Attribute 'TEST' to the reactions that have the same product as the test set reactions
    testset_products = df['products'].loc[df['Set'] == 'TEST'].unique()
    df['Set'].loc[df['products'].isin(testset_products)] = 'TEST'

    df.drop(columns=['Reaction_A>>C', 'isin_uspto_testset'], inplace=True)

    return df


def label_reactions_from_original_dataframe(df: pd.DataFrame, original_rxns: pd.Series, reaction_col: str):
    '''
    Creates a new column 'isin_original_set' and attributes True to the reactions that are in the original_rxns Series.

    --Inputs--
    df: pd.DataFrame, the dataframe containing the all the fictive reactions under the 'reaction_col' column
    original_rxns: pd.Series, the series containing the reactions that are in the original dataset
    reaction_col: str, the name of the column containing the reactions in df
    --Outputs--
    df: pd.DataFrame, the original dataframe containing the new 'isin_original_set' column, True for an element if the reaction is in the original dataset, False otherwise
    '''
    # format the reactions from A>B>C to A>>C
    df['Reaction_A>>C'] = df[reaction_col].apply(lambda x: x.split('>')[0] + '>>' + x.split('>')[2])

    # Determine which reactions are in teh original dataframe 
    df['isin_original_set'] = df['Reaction_A>>C'].isin(original_rxns)

    df.drop(columns=['Reaction_A>>C'], inplace=True)

    return df


def split_non_attributed_reactions(df: pd.DataFrame, train_split: float, seed: int=42):
    
    # check on train split < 1

    # Focus on only the reactions non attributed to a set already
    df_nolabel = df.loc[df['Set'] == '']

    # Group by product
    grouped = df_nolabel.groupby('products')

    # Shuffle the groups
    group_keys = list(grouped.groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)

    # Calculate splits
    total_groups = len(group_keys)
    train_size = int(train_split * total_groups)
    valid_size = int( (1-train_split)/2 * total_groups)
    
    # Split groups into train, validation, and test sets
    train_groups = group_keys[:train_size]
    valid_groups = group_keys[train_size:train_size+valid_size]
    test_groups = group_keys[train_size+valid_size:]

    # Assign splits to the DataFrame
    df.loc[df['products'].isin(train_groups), 'Set'] = 'TRAIN'
    df.loc[df['products'].isin(valid_groups), 'Set'] = 'VALID'
    df.loc[df['products'].isin(test_groups), 'Set'] = 'TEST'

    return df, df['Set'].value_counts()


def subset_n_random_reactions_per_template(full_df_conf, target, random_seed=42):
    random.seed(random_seed)
    df_concat = pd.DataFrame()
    for i in tqdm(full_df_conf['template_line'].unique()):
        df = full_df_conf[full_df_conf['template_line'] == i]
        len_df = len(df)
        if len_df >= target:
            ind = random.sample(range(0, len_df), target)
            df_concat = pd.concat([df_concat, df.iloc[ind]])
        else: # cannot be zero from the full_df_conf['template_line'].unique() list
            df_concat = pd.concat([df_concat, df])
    return df_concat

# Functions related to split the models and save the txt files

def calc_src_T1(df: pd.DataFrame, column_name: str)-> list:
    src_T1 = [df.at[i, column_name].split('>')[2] for i in range(len(df))]
    return src_T1


def calc_tgt_T1(df: pd.DataFrame, column_name: str)-> list:
    tgt_T1 = [df.at[i, column_name].split('>')[0] for i in range(len(df))]
    return tgt_T1


def calc_src_T2(df: pd.DataFrame, column_name: str)-> list:
    src_T2 = [df.at[i, column_name].split('>')[0] + '>>' + df.at[i, column_name].split('>')[2] for i in range(len(df))]
    return src_T2


def calc_tgt_T2(df: pd.DataFrame, column_name: str)-> list:
    tgt_T2 = [df.at[i, column_name].split('>')[1] for i in range(len(df))]
    return tgt_T2


def calc_src_T3(df: pd.DataFrame, column_name: str)-> list:
    src_T3 = [df.at[i, column_name].split('>')[0] + '>' + df.at[i, column_name].split('>')[1] for i in range(len(df))]
    return src_T3


def calc_tgt_T3(df: pd.DataFrame, column_name: str)-> list:
    tgt_T3 = [df.at[i, column_name].split('>')[2] for i in range(len(df))]
    return tgt_T3


def save_list_to_txt(path: str, list_to_save: list, filename: str):

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename, 'w') as f:
        for item in list_to_save:
            f.write("%s\n" % item)