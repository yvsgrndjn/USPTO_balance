
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


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def main(dataset_name, dataset_version, template_version, retro_reac, retro_template, Model_path_T2, Model_path_T3):

    reactions_conf_validation(dataset_name, dataset_version, template_version, retro_reac, retro_template, Model_path_T2, Model_path_T3)


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
        config['retro_template'],
        config['Model_path_T2'],
        config['Model_path_T3']        
    )