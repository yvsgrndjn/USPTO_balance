
import os
import pandas as pd
import yaml
import sys
import argparse
from tqdm import tqdm
from itertools import islice
import multiprocessing
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


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
    MappedReactions (list(str)):        List of mapped reactions in the format reactants(mapped)>>product(mapped)
    '''
    MappedReactions = list(singlestepretrosynthesis.rxn_mapper_batch.map_reactions(rxns_list))
    MappedReactions, preds_T2, rxns_list = remove_unmapped_rxns(MappedReactions, preds_T2, rxns_list)
    alt_taggedreactants = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = True, tag_reactants = True).split('>>')[0] for i in range(len(MappedReactions))]
    #alt_taggedproduct = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = True, tag_reactants = False).split('>>')[1] for i in range(len(MappedReactions))]
    reconstructed_rxns = [alt_taggedreactants[i] + '>' + preds_T2[i] for i in range(len(preds_T2))]
    reconstructed_rxns_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in reconstructed_rxns]

    ## Reconstruct the fully alternatively tagged reactions
    #alt_TaggedReactions = [alt_taggedreactants[i] + '>>' + alt_taggedproduct[i] for i in range(len(alt_taggedreactants))]

    ## Reconstruct the fully tagged reactions
    #taggedreactants = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = False, tag_reactants = True).split('>>')[0] for i in range(len(MappedReactions))]
    #taggedproduct = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = False, tag_reactants = False).split('>>')[1] for i in range(len(MappedReactions))]
    #TaggedReactions = [taggedreactants[i] + '>>' + taggedproduct[i] for i in range(len(taggedreactants))]
    return reconstructed_rxns_tok, MappedReactions#, alt_TaggedReactions, TaggedReactions


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
    rxns_list (list(str)):          List of reactions in the format reactants>>product 
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


def save_rxns_and_conf_to_csv(rxns_val: list, MappedReactions: list, conf_scores: list, dataset_name: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template: str):
    '''
    Saves the validated reactions and their confidence scores in a csv file.

    --Inputs--
    rxns_val (list(str)):           List of forward validated reactions
    MappedReactions (list(str)):    List of mapped reactions in the format reactants(mapped)>>product(mapped)
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
    df = pd.DataFrame({'rxns': rxns_val, 'mapped_rxns': MappedReactions, 'conf_scores': conf_scores})
    df.to_csv(f'{folder_path}/{name}.csv', index=False)

    temp_list.append(f'{folder_path}/{name}.csv')

    # Save the paths of saved subsets to a temp file to delete them once they are no longer needed
    save_created_files_to_temp_file(temp_path, temp_name, temp_list)
    print(f'Validated and saved {len(rxns_val)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template}')


def read_config(config_file: yaml):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    rxns_list                     = remove_incomplete_rxns(rxns_list)
    tok_rxns_list                 = tokenize_rxn_list(rxns_list)
    preds_T2                      = run_T2_predictions(tok_rxns_list, Model_path_T2, beam_size = 1, batch_size = 64, untokenize_output = True)
    rxns_T2_to_T3_tok, MappedReactions  = prepare_rxns_T2_for_T3(rxns_list, preds_T2)
    preds_T3, probs_T3            = run_T3_predictions(rxns_T2_to_T3_tok, Model_path_T3, beam_size = 3, batch_size = 64, untokenize_output = True)
    ind_match                     = find_ind_match_T3_preds_ref(preds_T3, rxns_list)
    rxns_list_with_reagents       = add_reagents_to_rxns_list(rxns_list, preds_T2, ind_match)
    MappedReactions_with_reagents = add_reagents_to_rxns_list(MappedReactions, preds_T2, ind_match)
    rxns_val, conf_scores         = keeps_val_rxns_and_scores(rxns_list_with_reagents, probs_T3, ind_match)
    save_rxns_and_conf_to_csv(rxns_val, MappedReactions_with_reagents, conf_scores, dataset_name, dataset_version, template_hash_version, retro_reac, retro_template)
    

def main(dataset_name: str, dataset_version: str, template_hash_version: str, retro_reac: str, retro_template: str, Model_path_T2: str, Model_path_T3: str):
    
    reactions_conf_validation(dataset_name, dataset_version, template_hash_version, retro_reac, retro_template, Model_path_T2, Model_path_T3)


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
        config['template_hash_version'],
        config['retro_reac'],
        config['retro_template'],
        config['Model_path_T2'],
        config['Model_path_T3']
        )
