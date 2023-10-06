
import os
import pandas as pd
import yaml
import sys
import argparse
from tqdm import tqdm
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


def load_rxns(GDB_version, template_version, retro_reac, retro_template):

    try:
        with open(f'/home/yves/Documents/GitHub/USPTO_balance/created_rxns_{GDB_version}_{template_version}/rxns_{retro_reac}_{retro_template}.txt', 'r') as f:
            rxns_list = []
            for line in f:
                rxns_list.append(line.split('\n')[0])
        return rxns_list
    
    except FileNotFoundError:
        print(f'No reactions found for retro_reac: {retro_reac} and retro_template: {retro_template}')
        return []


def tokenize_rxn_list(rxns_list):
    tok_rxns_list = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_list]
    return tok_rxns_list


def run_T2_predictions(SMILES_list, Model_path, beam_size: int = 1, batch_size: int = 64, untokenize_output:bool = True):
    [preds_T2, probs_T2] = singlestepretrosynthesis.Execute_Prediction(SMILES_list, Model_path, beam_size, batch_size, untokenize_output)
    return preds_T2[0]


def prepare_rxns_T2_for_T3(rxns_list, preds_T2):
    rxns_T2_list = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] + '>' + rxns_list[i].split('>>')[1] for i in range(len(preds_T2))]
    rxns_T2_to_T3 = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] for i in range(len(preds_T2))]
    rxns_T2_to_T3_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_T2_to_T3]
    return rxns_T2_list, rxns_T2_to_T3, rxns_T2_to_T3_tok


def run_T3_predictions(SMILES_list, Model_path, beam_size: int = 3, batch_size: int = 64, untokenize_output:bool = True):
    [preds_T3, probs_T3] = singlestepretrosynthesis.Execute_Prediction(SMILES_list, Model_path, beam_size, batch_size, untokenize_output)
    return preds_T3[0], probs_T3[0]


def find_ind_match_T3_preds_ref(preds_T3, rxns_list):
    #canonicalize preds_T3
    preds_T3 = [singlestepretrosynthesis.canonicalize_smiles(i) for i in preds_T3]

    preds_ref = [rxns_list[i].split('>>')[1] for i in range(len(rxns_list))]
    ind = [i for i in range(len(preds_T3)) if preds_T3[i] == preds_ref[i]]
    return ind


def keeps_match_confident_rxns(rxns_T2_list, probs_T3, ind_match, conf_score: int = 90):
    ind_keep = [probs_T3[i] > conf_score for i in range(len(probs_T3))]
    rxns_conf = [rxns_T2_list[i] for i in range(len(ind_keep)) if ind_keep[i] == True and i in ind_match]
    return rxns_conf


def save_conf_rxns(rxns_conf, GDB_version, template_version, retro_reac, retro_template):

    folder_path = f'saved_rxns_{GDB_version}_{template_version}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'/home/yves/Documents/GitHub/USPTO_balance/{folder_path}/rxns_{retro_reac}_{retro_template}.txt', 'w') as f:
        for item in rxns_conf:
            f.write(item + '\n')


def load_template_version(template_version):
    df_templates_split = pd.read_pickle(f'/home/yves/Documents/GitHub/USPTO_balance/data/templates_split/df_templates_to_enrich_part_{template_version}.pkl')
    return df_templates_split


def delete_evaluated_rxns(GDB_version, template_version, retro_reac, retro_template):

    name = f'rxns_{retro_reac}_{retro_template}'
    folder_path = f'./created_rxns_{GDB_version}_{template_version}'
    os.remove(f'{folder_path}/{name}.txt')


def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def reactions_conf_validation(GDB_version, template_version, retro_reac, retro_template, Model_path_T2, Model_path_T3):

    rxns_list = load_rxns(GDB_version, template_version, retro_reac, retro_template)

    if not rxns_list:
        return
    
    tok_rxns_list = tokenize_rxn_list(rxns_list)
    preds_T2 = run_T2_predictions(tok_rxns_list, Model_path_T2, beam_size = 1, batch_size = 64, untokenize_output = True)
    rxns_T2_list, rxns_T2_to_T3, rxns_T2_to_T3_tok = prepare_rxns_T2_for_T3(rxns_list, preds_T2)
    preds_T3, probs_T3 = run_T3_predictions(rxns_T2_to_T3_tok, Model_path_T3, beam_size = 3, batch_size = 64, untokenize_output = True)
    ind_match = find_ind_match_T3_preds_ref(preds_T3, rxns_list)
    rxns_conf = keeps_match_confident_rxns(rxns_T2_to_T3, probs_T3, ind_match, conf_score = 0.9)
    save_conf_rxns(rxns_conf, GDB_version, template_version, retro_reac, retro_template)
    #delete_evaluated_rxns(GDB_version, template_version, retro_reac, retro_template)

def main(GDB_version, template_version, Model_path_T2, Model_path_T3):

    df_templates_split = load_template_version(template_version)

    for retro_reac, retro_template in tqdm(zip(df_templates_split['retro_reac'], df_templates_split['retro_templates'])):
        reactions_conf_validation(GDB_version, template_version, retro_reac, retro_template, Model_path_T2, Model_path_T3)


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
        config['GDB_version'],
        config['template_version'],
        config['Model_path_T2'],
        config['Model_path_T3']        
    )