import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import pickle
from itertools import chain
from itertools import islice
import yaml
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()

from uspto_balance.C_part1_framework import main as c_part1_framework
from uspto_balance.D_part2_framework import main as d_part2_framework
from uspto_balance.E_part3_framework import main as e_part3_framework

#Module containing all functions related to the balancing dataset workflow

# Utilities functions ------------------------------


def canonicalize(smiles):
    '''
    Converts a smile string into a rdkit canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


#C_part1_framework specific functions ------------------------------


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


#D_part2_framework functions ------------------------------


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


#E_part3_framework functions ------------------------------


def load_rxns(dataset_name, dataset_version, template_version, retro_reac, retro_template, rxns_number = 10000):
    '''
    Loads the rxns list that were created in part 2 of the dataset subset matching the retro_reac pattern and applying retro_template to it
    '''
    try:
        folder_path = f'./results/created_rxns/{dataset_name}'
        name        = f'rxns_{dataset_version}_{retro_reac}_{retro_template}' 

        with open(f'{folder_path}/{name}.txt', 'r') as f:
            rxns_list = []
            rxns_list = list(islice(f, rxns_number))
            rxns_list = [rxns_list[i].split('\n')[0] for i in range(len(rxns_list))]

        return rxns_list
    
    except FileNotFoundError:
        print(f'No reactions found for retro_reac: {retro_reac} and retro_template: {retro_template}')

        return []


def tokenize_rxn_list(rxns_list):
    '''
    Tokenizes a list of reactions
    '''
    tok_rxns_list = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_list]

    return tok_rxns_list


def run_T2_predictions(tok_rxns_list, Model_path, beam_size: int = 1, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>>product) and predicts the needed reagents for the reaction to take place.
    Gives back the list of reagents that has the same length as the input SMILES list.
    '''
    [preds_T2, probs_T2] = singlestepretrosynthesis.Execute_Prediction(tok_rxns_list, Model_path, beam_size, batch_size, untokenize_output)
    
    return preds_T2[0]


def prepare_rxns_T2_for_T3(rxns_list, preds_T2):
    '''
    From the rxns_list and the predicted reagents for each reaction, returns a list of tokenized reactions
    in an appropriate format to use as input to forward tag T3
    reactants tagged "!" > reagents (tokenized)
    '''
    MappedReactions = list(singlestepretrosynthesis.rxn_mapper_batch.map_reactions(rxns_list))
    taggedreactants = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = True, tag_reactants = True).split('>>')[0] for i in range(len(MappedReactions))]
    reconstructed_rxns = [taggedreactants[i] + '>' + preds_T2[i] for i in range(len(preds_T2))]
    reconstructed_rxns_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in reconstructed_rxns]
    
    return reconstructed_rxns_tok


def run_T3_predictions(rxns_T2_to_T3_tok, Model_path, beam_size: int = 3, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>reagents) and performs forward prediction on them.
    Gives back the list of predicted products that has the same length as the input SMILES list.    
    '''
    [preds_T3, probs_T3] = singlestepretrosynthesis.Execute_Prediction(rxns_T2_to_T3_tok, Model_path, beam_size, batch_size, untokenize_output)
    
    return preds_T3[0], probs_T3[0]


def remove_unmapped_rxns(MappedReactions, preds_T2, rxns_list):
    indices_to_remove = [i for i, value in enumerate(MappedReactions) if value == '>>']

    for index in reversed(indices_to_remove):
        del MappedReactions[index]
        del preds_T2[index]
        del rxns_list[index]
    return MappedReactions, preds_T2, rxns_list


def find_ind_match_T3_preds_ref(preds_T3, rxns_list):
    '''
    Takes as input the forward predictions of T3 and the original reactions list, returns the indices of the predicted products matching the original products (returns indices of forward validated reactions)
    '''
    preds_T3 = [singlestepretrosynthesis.canonicalize_smiles(i) for i in preds_T3]

    preds_ref = [rxns_list[i].split('>>')[1] for i in range(len(rxns_list))]
    ind = [i for i in range(len(preds_T3)) if preds_T3[i] == preds_ref[i]]
    
    return ind


def keeps_match_confident_rxns(rxns_list, probs_T3, ind_match, conf_score = 0.9):
    '''
    
    '''
    ind_keep = [probs_T3[i] > conf_score for i in range(len(probs_T3))]
    rxns_conf = [rxns_list[i] for i in range(len(ind_keep)) if ind_keep[i] == True and i in ind_match]
    
    return rxns_conf


def save_conf_rxns(rxns_conf, dataset_name, dataset_version, template_version, retro_reac, retro_template):

    retro_template = retro_template.replace('/', 'slash')

    folder_path = f'./results/saved_rxns/{dataset_name}'
    name        = f'rxns_{dataset_version}_{retro_reac}_{retro_template}'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'{folder_path}/{name}.txt', 'w') as f:
        for item in rxns_conf:
            f.write(item + '\n')

    print(f'Validated and saved {len(rxns_conf)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')


def delete_evaluated_rxns(dataset_name, dataset_version, template_version, retro_reac, retro_template):

    retro_template = retro_template.replace('/', 'slash')

    folder_path = f'./results/created_rxns/{dataset_name}'
    name = f'rxns_{dataset_version}_{retro_reac}_{retro_template}'
    
    os.remove(f'{folder_path}/{name}.txt')


def reactions_conf_validation(dataset_name, dataset_version, template_version, retro_reac, retro_template, Model_path_T2, Model_path_T3):

    rxns_list = load_rxns(dataset_name, dataset_version, template_version, retro_reac, retro_template)

    if not rxns_list:
        return
    
    tok_rxns_list = tokenize_rxn_list(rxns_list)
    preds_T2 = run_T2_predictions(tok_rxns_list, Model_path_T2, beam_size = 1, batch_size = 64, untokenize_output = True)
    rxns_T2_to_T3_tok = prepare_rxns_T2_for_T3(rxns_list, preds_T2)
    preds_T3, probs_T3 = run_T3_predictions(rxns_T2_to_T3_tok, Model_path_T3, beam_size = 3, batch_size = 64, untokenize_output = True)
    ind_match = find_ind_match_T3_preds_ref(preds_T3, rxns_list)
    rxns_conf = keeps_match_confident_rxns(rxns_list, probs_T3, ind_match, conf_score = 0.9)
    save_conf_rxns(rxns_conf, dataset_name, dataset_version, template_version, retro_reac, retro_template)
    #delete_evaluated_rxns(dataset_name, dataset_version, template_version, retro_reac, retro_template)


#dataset_equilibration functions ------------------------------


def extract_subset_from_dataset(dataset_name, dataset_version, retro_reac, retro_template, path_to_folder):
    
    #initialize config variables
    template_version           = f"{retro_reac}"
    dataset_path               = f"{path_to_folder}data/{dataset_name}_{dataset_version}.txt"

    #check if part 1 needs to be run or not
    folder_path = f'./results/datasets/{dataset_name}'
    name        = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'

    #only run part 1 if the subset does not exist yet
    if not os.path.exists(f'{folder_path}/{name}.txt'):

        #replace the ';' characters in the config file name to avoid errors
        config_file_path = f'./config_files/config_part1_{template_version}.yaml'.replace(';','')

        #create config file for part 1
        with open(f'{config_file_path}', 'w') as f:

            f.write(f'dataset_name: "{dataset_name}"\n')
            f.write(f'dataset_path: "{path_to_folder}data/{dataset_name}_{dataset_version}.txt"\n')
            f.write(f'dataset_version: "{dataset_version}"\n')
            f.write(f'template_version: "{template_version}"\n')
            f.write(f'retro_reac: "{retro_reac}"\n')
            f.write(f'retro_template: "{retro_template}"\n')

        # run part 1
        with open(f'{config_file_path}', 'r') as f:
            config1 = yaml.safe_load(f)

        c_part1_framework(          
            config1['dataset_name'],
            config1['dataset_path'],
            config1['dataset_version'],
            config1['template_version'],
            config1['retro_reac'],
            config1['retro_template']
            )


def create_reactions_using_template(dataset_name, dataset_version, retro_reac, retro_template, path_to_folder):
    
    #initialize config variables
    template_version           = f"{retro_reac}"

    #replace the ';' characters in the config file name to avoid errors
    config_file_path2 = f'{path_to_folder}config_files/config_part2_{template_version}.yaml'.replace(';','')

    #create config file for part 2
    with open(f'{config_file_path2}', 'w') as f:
        f.write(f'dataset_name: "{dataset_name}"\n')
        f.write(f'dataset_version: "{dataset_version}"\n')
        f.write(f'template_version: "{template_version}"\n')
        f.write(f'retro_reac: "{retro_reac}"\n')
        f.write(f'retro_template: "{retro_template}"\n')

    # run part 2
    with open(f'{config_file_path2}', 'r') as f:
        config2 = yaml.safe_load(f)
    
    d_part2_framework(
        config2['dataset_name'],
        config2['dataset_version'],
        config2['template_version'],
        config2['retro_reac'],
        config2['retro_template']
    )


def validate_created_reactions(dataset_name, dataset_version, retro_reac, retro_template, path_to_folder, path_models):
    
    #initialize config variables
    template_version           = f"{retro_reac}"
    Model_path_T2              = f"{path_models}USPTO_STEREO_separated_T2_Reagent_Pred_225000.pt"
    Model_path_T3              = f"{path_models}T3_Fwd_Tag_model_step_300000.pt"
       
    config_file_path3 = f'{path_to_folder}config_files/config_part3_{template_version}.yaml'.replace(';','')

    #create config file for part 3
    with open(f'{config_file_path3}', 'w') as f:
        f.write(f'dataset_name: "{dataset_name}"\n')
        f.write(f'dataset_version: "{dataset_version}"\n')
        f.write(f'template_version: "{template_version}"\n')
        f.write(f'retro_reac: "{retro_reac}"\n')
        f.write(f'retro_template: "{retro_template}"\n')
        f.write(f'Model_path_T2: {Model_path_T2}\n')
        f.write(f'Model_path_T3: {Model_path_T3}')

    # run part 3
    with open(f'{config_file_path3}', 'r') as f:
        config3 = yaml.safe_load(f)
    
    e_part3_framework(
        config3['dataset_name'],
        config3['dataset_version'],
        config3['template_version'],
        config3['retro_reac'],
        config3['retro_template'],
        config3['Model_path_T2'],
        config3['Model_path_T3']
    )


def get_txt_file(path_to_file):
    with open(path_to_file, 'r') as f:
        file = []
        for line in f:
            file.append(line.split('\n')[0])
    return file


def append_list_to_file(path_to_file, list_to_append):
    with open(path_to_file, 'a') as f:
        for item in list_to_append:
            f.write(item + '\n')


def append_n_elements_to_file(path_to_file, list_to_append, n_elements):
    with open(path_to_file, 'a') as f:
        for i in range(n_elements):
            f.write(list_to_append[i] + '\n')


def append_saved_rxns_until_enrichment_target(dataset_name, dataset_version, retro_reac, retro_template, template_frequency, frequency_target: int = 10000):

    retro_template = retro_template.replace('/', 'slash')
    folder_path  = f'./results/saved_rxns/{dataset_name}'
    name         = f'rxns_{dataset_version}_{retro_reac}_{retro_template}'
    name_to_save = f'rxns_{retro_reac}_{retro_template}'
    
    if os.path.exists(f'{folder_path}/{name}.txt'):
        saved_rxns = get_txt_file(f'{folder_path}/{name}.txt')
        template_frequency += len(saved_rxns)

        if template_frequency < frequency_target:
            append_list_to_file(f'{folder_path}/full_{name_to_save}.txt', saved_rxns)

        else:
            how_many_to_append = frequency_target - (template_frequency - len(saved_rxns))
            append_n_elements_to_file(f'{folder_path}/full_{name_to_save}.txt', saved_rxns, how_many_to_append)

    return template_frequency


#Functions used in other steps than the actual dataset equilibration ------------------------------


def select_templates_to_enrich(data: pd.Series, min_number: int = 10, target_number: int = 10000):
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









