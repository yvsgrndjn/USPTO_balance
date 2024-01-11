#imports 
import os
import yaml
import argparse
import sys
import shutil
import pandas as pd

#module imports
from uspto_balance.C_part1_framework import main as c_part1_framework
from uspto_balance.D_part2_framework import main as d_part2_framework
from uspto_balance.E_part3_framework import main as e_part3_framework

print('Imports done') 

#functions definition-------


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_subset_from_dataset(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder):
    print('In extract_subset_from_dataset')

    #initialize config variables

    #template_version           = f"{retro_reac}".replace('/', 'slash')
    template_hash_version      = f"{template_hash}_{template_line}" #new-------
    dataset_path               = f"{path_to_folder}data/{dataset_name}_{dataset_version}.txt"

    #check if part 1 needs to be run or not
    folder_path = f'./results/datasets/{dataset_name}'
    
    #name        = f'{dataset_name}_sub_{dataset_version}_{template_version}'
    name        = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}' #new-------

    #only run part 1 if the subset does not exist yet
    if not os.path.exists(f'{folder_path}/{name}.txt'):

        #replace the ';' characters in the config file name to avoid errors
        
        #config_file_path = f'./config_files/config_part1_{dataset_name}_{template_version}.yaml'.replace(';','')
        config_file_path = f'./config_files/config_part1_{dataset_name}_{template_hash_version}.yaml'.replace(';','') #new-------

        #create config file for part 1
        with open(f'{config_file_path}', 'w') as f:

            f.write(f'dataset_name: \'{dataset_name}\'\n')
            f.write(f'dataset_path: \'{dataset_path}\'\n')
            f.write(f'dataset_version: \'{dataset_version}\'\n')
            #f.write(f'template_version: \'{template_version}\'\n')
            f.write(f'template_hash_version: \'{template_hash_version}\'\n') #new-------
            f.write(f'retro_reac: \'{retro_reac}\'\n')
            f.write(f'retro_template: \'{retro_template}\'\n')

        # run part 1
        with open(f'{config_file_path}', 'r') as f:
            config1 = yaml.safe_load(f)

        c_part1_framework(          
            config1['dataset_name'],
            config1['dataset_path'],
            config1['dataset_version'],
            #config1['template_version'],
            config1['template_hash_version'], #new-------
            config1['retro_reac'],
            config1['retro_template']
            )


def create_reactions_using_template(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder):
    print('In create_reactions_using_template')

    #initialize config variables
    #template_version           = f"{retro_reac}".replace('/', 'slash')
    template_hash_version      = f"{template_hash}_{template_line}" #new-------

    #replace the ';' characters in the config file name to avoid errors
    
    #config_file_path2 = f'{path_to_folder}config_files/config_part2_{dataset_name}_{template_version}.yaml'.replace(';','')
    config_file_path2 = f'{path_to_folder}config_files/config_part2_{dataset_name}_{template_hash_version}.yaml'.replace(';','') #new-------

    #create config file for part 2
    with open(f'{config_file_path2}', 'w') as f:
        #f.write(f'dataset_name: "{dataset_name}"\n')
        #f.write(f'dataset_version: "{dataset_version}"\n')
        #f.write(f'template_version: "{template_version}"\n')
        #f.write(f'retro_reac: "{retro_reac}"\n')
        #f.write(f'retro_template: "{retro_template}"\n')
        f.write(f'dataset_name: \'{dataset_name}\'\n')
        f.write(f'dataset_version: \'{dataset_version}\'\n')
        #f.write(f'template_version: \'{template_version}\'\n')
        f.write(f'template_hash_version: \'{template_hash_version}\'\n') #new-------
        f.write(f'retro_reac: \'{retro_reac}\'\n')
        f.write(f'retro_template: \'{retro_template}\'\n')


    # run part 2
    with open(f'{config_file_path2}', 'r') as f:
        config2 = yaml.safe_load(f)
    
    d_part2_framework(
        config2['dataset_name'],
        config2['dataset_version'],
        #config2['template_version'],
        config2['template_hash_version'], #new-------
        config2['retro_reac'],
        config2['retro_template']
    )


def validate_created_reactions(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder, path_models):
    print('In validate_created_reactions')
    #initialize config variables

    #template_version           = f"{retro_reac}".replace('/', 'slash')
    template_hash_version      = f"{template_hash}_{template_line}" #new-------
    Model_path_T2              = f"{path_models}USPTO_STEREO_separated_T2_Reagent_Pred_225000.pt"
    Model_path_T3              = f"{path_models}T3_Fwd_Tag_model_step_300000.pt"
       
    #config_file_path3 = f'{path_to_folder}config_files/config_part3_{dataset_name}_{template_version}.yaml'.replace(';','')
    config_file_path3 = f'{path_to_folder}config_files/config_part3_{dataset_name}_{template_hash_version}.yaml'.replace(';','')

    #create config file for part 3
    with open(f'{config_file_path3}', 'w') as f:
        #f.write(f'dataset_name: "{dataset_name}"\n')
        #f.write(f'dataset_version: "{dataset_version}"\n')
        #f.write(f'template_version: "{template_version}"\n')
        #f.write(f'retro_reac: "{retro_reac}"\n')
        #f.write(f'retro_template: "{retro_template}"\n')
        #f.write(f'Model_path_T2: {Model_path_T2}\n')
        #f.write(f'Model_path_T3: {Model_path_T3}')
        f.write(f'dataset_name: \'{dataset_name}\'\n')
        f.write(f'dataset_version: \'{dataset_version}\'\n')
        #f.write(f'template_version: \'{template_version}\'\n')
        f.write(f'template_hash_version: \'{template_hash_version}\'\n')
        f.write(f'retro_reac: \'{retro_reac}\'\n')
        f.write(f'retro_template: \'{retro_template}\'\n')
        f.write(f'Model_path_T2: {Model_path_T2}\n')
        f.write(f'Model_path_T3: {Model_path_T3}')

    # run part 3
    with open(f'{config_file_path3}', 'r') as f:
        config3 = yaml.safe_load(f)
    
    e_part3_framework(
        config3['dataset_name'],
        config3['dataset_version'],
        #config3['template_version'],
        config3['template_hash_version'], #new-------
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


def append_df_to_csv(path_to_csv, df_to_append):
    df_to_append.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index = False)    


def append_n_elements_to_file(path_to_file, list_to_append, n_elements):
    with open(path_to_file, 'a') as f:
        for i in range(n_elements):
            f.write(list_to_append[i] + '\n')


def delete_dataset_subsets(dataset_name, dataset_version, template_hash_version, folder_path):
    '''
    Delete the .txt and .pkl subsets of the dataset created in C_part1_framework.py
    to free some memory (some datasets might be recalculated several times)
    '''
    folder_path     = f'./results/datasets/{dataset_name}'
    folder_path_mol = f'./results/datasets/{dataset_name}_mol'
    #name            = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'
    os.remove(f'{folder_path}/{name}.txt')
    os.remove(f'{folder_path_mol}/{name}.pkl')


def delete_all_files_from_folder(path, dataset_name):
    folder_path = f'{path}/{dataset_name}'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
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


def append_saved_rxns_until_enrichment_target(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, template_frequency, frequency_target: int = 10000):
    print('In append_saved_rxns_until_enrichment_target')

    retro_reac      = retro_reac.replace('/', 'slash')
    retro_template  = retro_template.replace('/', 'slash')
    folder_path     = f'./results/saved_rxns/{dataset_name}'
    #name           = f'{dataset_name}_sub_{dataset_version}_{retro_reac}'
    template_hash_version      = f"{template_hash}_{template_line}" #new-------
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'
    #name_to_save    = f'{dataset_name}_{retro_reac}_{retro_template}'
    name_to_save    = f'{dataset_name}_{template_hash_version}'

    if os.path.exists(f'{folder_path}/{name}.txt'):
        saved_rxns = get_txt_file(f'{folder_path}/{name}.txt')
        template_frequency += len(saved_rxns)

        if template_frequency < frequency_target:
            append_list_to_file(f'{folder_path}/full_{name_to_save}.txt', saved_rxns)
            
        else:
            how_many_to_append = frequency_target - (template_frequency - len(saved_rxns))
            append_n_elements_to_file(f'{folder_path}/full_{name_to_save}.txt', saved_rxns, how_many_to_append)
            template_frequency = template_frequency - len(saved_rxns) + how_many_to_append

    else:
        print(f'No reactions found under: {folder_path}/{name}.txt (append_saved_rxns_until_enrichment_target)')

    #remove the created subsets to avoid using memory space
    if os.path.exists(f'{folder_path}/{name}.txt'):
        #delete_dataset_subsets(dataset_name, dataset_version, retro_reac, folder_path)
        delete_dataset_subsets(dataset_name, dataset_version, template_hash_version, folder_path)

    return template_frequency


def append_saved_rxns_until_enrichment_target_pkl(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, template_frequency, frequency_target: int = 10000):
    print('In append_saved_rxns_until_enrichment_target')

    retro_reac      = retro_reac.replace('/', 'slash')
    retro_template  = retro_template.replace('/', 'slash')
    folder_path     = f'./results/saved_rxns/{dataset_name}'
    template_hash_version      = f"{template_hash}_{template_line}" #new-------
    name            = f'{dataset_name}_sub_{dataset_version}_{template_hash_version}'
    name_to_save    = f'{dataset_name}_{template_hash_version}'

    if os.path.exists(f'{folder_path}/{name}.csv'):
        #saved_rxns = get_txt_file(f'{folder_path}/{name}.txt')
        df_saved_rxns = pd.read_csv(f'{folder_path}/{name}.csv')
        numel_conf = sum(i > 0.95 for i in df_saved_rxns['conf_scores'].values.tolist())
        template_frequency += numel_conf
        num_added_rxns = len(df_saved_rxns)
        
        append_df_to_csv(f'{folder_path}/full_{name_to_save}.csv', df_saved_rxns)

    else:
        print(f'No reactions found under: {folder_path}/{name}.csv (append_saved_rxns_until_enrichment_target)')

    #remove the created subsets to avoid using memory space
    if os.path.exists(f'{folder_path}/{name}.csv'):
        delete_dataset_subsets(dataset_name, dataset_version, template_hash_version, folder_path)

    return template_frequency, num_added_rxns


# main definition    -------
def main(dataset_name, retro_reac, retro_template, template_hash, template_line, path_to_folder, path_models, template_frequency, frequency_target: int = 10000):
    print('In main')
    counter = 1
    num_added_rxns = 0
    num_added_rxns_before = 0

    initial_template_frequency = template_frequency

    while template_frequency < frequency_target and counter <= 100:
        print(f'Iteration {counter}')
        template_frequency_before = template_frequency
        num_added_rxns_before += num_added_rxns
        extract_subset_from_dataset(dataset_name, counter, retro_reac, retro_template, template_hash, template_line, path_to_folder)
        create_reactions_using_template(dataset_name, counter, retro_reac, retro_template, template_hash, template_line, path_to_folder)
        validate_created_reactions(dataset_name, counter, retro_reac, retro_template, template_hash, template_line, path_to_folder, path_models)
        #template_frequency = append_saved_rxns_until_enrichment_target(dataset_name, counter, retro_reac, retro_template, template_hash, template_line, template_frequency, frequency_target)
        template_frequency, num_added_rxns = append_saved_rxns_until_enrichment_target_pkl(dataset_name, counter, retro_reac, retro_template, template_hash, template_line, template_frequency, frequency_target)
        counter += 1

        added_reactions = template_frequency - template_frequency_before
        print(f'Validated {num_added_rxns} reactions out of which {added_reactions} are confident > 0.95,  reactions added to the actual total {num_added_rxns_before} (total of validated and confident reactions = {template_frequency} / {frequency_target})for retro_reac: {retro_reac} and retro_template: {retro_template}')

    print(f'Enrichment finished (with counter = {counter}):  initial {initial_template_frequency} reactions were enriched to {template_frequency} for retro_reac: {retro_reac} and retro_template: {retro_template}')
    
    print('Delete remaining files...')
    delete_all_files_from_folder(f'./results/datasets', dataset_name)
    delete_all_files_from_folder(f'./results/datasets', f'{dataset_name}_mol')
    delete_all_files_from_folder(f'./results/created_rxns', dataset_name)
    delete_all_files_from_folder(f'./results/saved_rxns', dataset_name)
    delete_all_files_from_folder('./config_files', '')
    print('Done')

def main_balance():
    print('In main_balance')
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
        config['retro_reac'],
        config['retro_template'],
        config['template_hash'],#new---------
        config['template_line'],#new---------
        config['path_to_folder'],
        config['path_models'], 
        config['template_frequency'],
        config['frequency_target']
        )


if __name__ == '__main__':
    main_balance()
