import os
import yaml
import argparse
import sys
import shutil
import pandas as pd
import random

from uspto_balance.C_part1_framework import main as c_part1_framework
from uspto_balance.D_part2_framework import main as d_part2_framework
from uspto_balance.E_part3_framework import main as e_part3_framework

print('Imports done') 


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


def main(dataset_name, retro_reac, retro_template, template_hash, template_line, path_to_folder, path_models, template_frequency, frequency_target: int = 10000):
    '''
    Main function to run the enrichment of a given dataset_name, retro_reac, retro_template, template_hash, and template_line from 'template_frequency' until the desired number of reactions 'frequency_target'
    is reached. For a given reaction retrosynthetic template ('retro_template'), the function will create similar fictive reactions that are forward-validated by a retrosynthesis prediction model (TTL).
    The dataset 'dataset_name' is used as a pool of molecules on which the template 'retro_template' is applied to create fictive reactions (of the form reactant(s)>>product). Once the reactions are created, they are passed through the reagent prediction (T2)
    model of the TTL (to give a format: reactant(s)>reagent(s)>product), and the reactions will be used as input in the forward validation (T3-FT) model of the TTL, that will act as a filter on two different levels:
        1) The model will be presented reactant(s)>reagent(s) and will predict what the product of such reaction should be. If the predicted product is the same as the actual product, the reaction is considered as validated.
        2) The forward validation model will also give a confidence score for each reaction, and only the reactions with a confidence score above 0.95 will be considered as confident enough to be kept.
    All the validated reactions (not only the validated and confident enough reactions) are saved under a csv file. The function will run until the number of validated and confident enough reactions reaches the 'frequency_target' or there are no more molecules in the dataset to create new reactions.

    --Inputs--
    dataset_name (str):           name of the dataset (str) ex: GDB13S, USPTO. Prerequisite (for the module, not the function): The dataset is divided in 1000 different parts
    retro_reac (str):             SMARTS pattern of the substructure to match
    retro_template (str):         retrosynthetic reaction template in SMART format
    template_hash (str):          hash of the retrosynthetic reaction template
    template_line (str(int)):     line of the retrosynthetic reaction template in the dataframe containing all the templates to be enriched
    path_to_folder (str):         path to the folder containing the config_files and the data folder
    path_models (str):            path to the folder containing the models
    template_frequency (int):     Actual number of reactions that for which the extracted template corresponds to 'retro_template'
    frequency_target (int):       Enrichment target for the number of reactions that will show a reactivity corresponding to 'retro_template'

    --Returns--
    None, but saves the validated reactions under a csv file
    '''
    print('In main')
    counter = 1
    num_added_rxns = 0
    num_added_rxns_before = 0

    # Initialize random seed to shuffle the order of the dataset iterations
    index_list = list(range(1, 1001))
    random.seed(template_line)
    random.shuffle(index_list)

    initial_template_frequency = template_frequency

    while template_frequency < frequency_target and counter <= 1000:
        print(f'Iteration {counter}')

        # Keep track of the number of reactions and template frequency evolution
        template_frequency_before = template_frequency
        num_added_rxns_before += num_added_rxns

        # Define the dataset version
        dataset_version = index_list[counter-1] 

        # Run the enrichment on the decided dataset version
        extract_subset_from_dataset(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder)
        create_reactions_using_template(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder) 
        validate_created_reactions(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, path_to_folder, path_models) 
        template_frequency, num_added_rxns = append_saved_rxns_until_enrichment_target_csv(dataset_name, dataset_version, retro_reac, retro_template, template_hash, template_line, template_frequency, frequency_target)
        
        counter += 1
        added_reactions = template_frequency - template_frequency_before
        print(f'Validated {num_added_rxns} reactions out of which {added_reactions} are confident > 0.95,  reactions added to the actual total {num_added_rxns_before} (total of validated and confident reactions = {template_frequency} / {frequency_target})for retro_reac: {retro_reac} and retro_template: {retro_template}')

    # Append number of enrichment runs information to the results csv file
    add_dataset_fraction_to_csv(dataset_name, dataset_version, template_hash, template_line, retro_reac, retro_template)
    print(f'Enrichment finished (with counter = {counter-1}):  initial {initial_template_frequency} reactions were enriched to {template_frequency} for retro_reac: {retro_reac} and retro_template: {retro_template}')
    

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
        config['template_hash'],
        config['template_line'],
        config['path_to_folder'],
        config['path_models'], 
        config['template_frequency'],
        config['frequency_target']
        )


if __name__ == '__main__':
    main_balance()
