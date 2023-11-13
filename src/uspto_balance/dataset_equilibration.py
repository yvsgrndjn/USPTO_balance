#imports 
import os
import yaml
import argparse
import sys


#module imports
from C_part1_framework import main as c_part1_framework
from D_part2_framework import main as d_part2_framework
from E_part3_framework import main as e_part3_framework

#functions definition-------


def read_config(config_file):
    '''
    Reads the yaml config_file to extract the arguments for the main function
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


# main definition    -------
def main(dataset_name, retro_reac, retro_template, path_to_folder, path_models, template_frequency, frequency_target: int = 10000):

    counter = 0
    initial_template_frequency = template_frequency

    while template_frequency < frequency_target and counter <= 100:
        
        counter += 1

        extract_subset_from_dataset(dataset_name, counter, retro_reac, retro_template, path_to_folder)
        create_reactions_using_template(dataset_name, counter, retro_reac, retro_template, path_to_folder)
        validate_created_reactions(dataset_name, counter, retro_reac, retro_template, path_to_folder, path_models)
        template_frequency = append_saved_rxns_until_enrichment_target(dataset_name, counter, retro_reac, retro_template, template_frequency, frequency_target)

    print(f'Enriched initial {initial_template_frequency} reactions to {template_frequency} for retro_reac: {retro_reac} and retro_template: {retro_template}')


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
        config['retro_reac'],
        config['retro_template'],
        config['path_to_folder'],
        config['path_models'], 
        config['template_frequency'],
        config['frequency_target']

