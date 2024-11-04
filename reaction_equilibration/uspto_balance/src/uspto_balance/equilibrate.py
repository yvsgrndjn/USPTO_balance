import os
import time
import yaml
import random
import argparse
import pandas as pd

from src.uspto_balance.find_matches_and_format import FindMatchesApplyTemplateFormatReactions
from src.uspto_balance.validate_reactions_TTL import ValidateReactionsWithTTL

print('Imports done')

class Equilibrate:
    def __init__(self, dataset_name:str, retro_reac:str, retro_template:str, template_line:int, template_frequency:int, frequency_target:int, T2_model_path:str, T3_model_path:str, run_from_path:str='./'):
        self.dataset_name       = dataset_name
        self.retro_reac         = retro_reac
        self.retro_template     = retro_template
        self.template_line      = template_line
        self.initial_template_frequency = template_frequency
        self.frequency_target   = frequency_target
        self.T2_model_path      = T2_model_path
        self.T3_model_path      = T3_model_path
        self.folder_path        = run_from_path

        self.stats_file_path    = f'{self.folder_path}results/saved_rxns/{self.dataset_name}/full_{self.dataset_name}_stats_template_{template_line}.csv'
        self.results_csv_folder_path = f'{self.folder_path}results/saved_rxns/{self.dataset_name}/'
        self.results_csv_path   = f'{self.results_csv_folder_path}full_{self.dataset_name}_template_{template_line}.csv'
        self.counter            = 0
        self.num_matches        = 0
        self.num_created_rxns   = 0
        self.num_val_rxns       = 0
        self.num_val_conf_rxns  = 0
        self.time_elapsed       = 0
        self.initial_template_frequency = 0
        self.cpu_type           = os.popen('lscpu').read().split('\n')[4].split(':')[1].strip()

    def get_checkpoint_info(self):
        #check for files existence
        try:
            df_checkpoint = pd.read_csv(self.stats_file_path)
            self.counter                    = df_checkpoint.at[0, 'dataset fractions']
            self.num_created_rxns           = df_checkpoint.at[0, 'created_rxns']
            self.num_val_rxns               = df_checkpoint.at[0, 'validated reactions']
            self.num_val_conf_rxns          = df_checkpoint.at[0, 'validated and confident reactions']
            self.time_elapsed               = df_checkpoint.at[0, 'time elapsed']
            num_added_rxns_so_far           = self.num_val_conf_rxns - self.initial_template_frequency
            print(f'Continue enrichment from fraction {self.counter} with so far {self.num_val_conf_rxns} validated and confident reactions (out of target {self.frequency_target}), starting from originally {self.initial_template_frequency} reactions')
        except FileNotFoundError:
            return

    def initialize_random_indices(self):   
        self.index_list = list(range(1, 1001))
        random.seed(self.template_line)
        random.shuffle(self.index_list)

    def get_dataset_version(self):
        return self.index_list[self.counter]

    def create_folder_if_not_exists(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def append_df_to_csv(self, path_to_csv: str, df_to_append: pd.DataFrame):
        '''
        Appends a dataframe to a csv file

        --Inputs--
        path_to_csv (str):           path to the csv file to be appended
        df_to_append (pd.DataFrame): dataframe to be appended to the csv file

        --Returns--
        None, appends the dataframe to the csv file
        '''
        df_to_append.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index = False)

    def update_checkpoint(self):
        time_elapsed = time.time() - self.start
        self.dataset_version = self.get_dataset_version()
        fraction = self.index_list.index(self.dataset_version)
        df = pd.DataFrame({'template line'                  :self.template_line,
                        'dataset fractions'                 : [fraction],
                        'created_rxns'                      : [self.num_created_rxns],
                        'validated reactions'               : [self.num_val_rxns],
                        'validated and confident reactions' : [self.num_val_conf_rxns],
                        'time elapsed'                      : [time_elapsed],
                        'cpu type'                          : self.cpu_type})
        df.to_csv(self.stats_file_path, index=False)
        self.stats = df

    def process(self):
        # Load checkpoint data if it exists
        self.get_checkpoint_info()

        # Initialize 1000 random indices
        self.initialize_random_indices()
        
        self.start = time.time()

        while self.num_val_conf_rxns < self.frequency_target and self.counter < 1000:
            print(f'Iteration {self.counter}/999')        
            
            self.dataset_version = self.get_dataset_version()

            # Initialize the class to match molecules and apply templates
            create_fictive_rxns = FindMatchesApplyTemplateFormatReactions(self.dataset_name, 
                                                                          self.dataset_version, 
                                                                          self.retro_reac, 
                                                                          self.retro_template, 
                                                                          self.template_line, 
                                                                          self.folder_path)
            # create the fictive reactions and count them
            fictive_rxns = create_fictive_rxns.main()
            self.num_created_rxns += len(fictive_rxns)

            # Initialize the class to validate reactions with the TTL model
            validate_reactions = ValidateReactionsWithTTL(fictive_rxns, self.T2_model_path, self.T3_model_path)
            val_rxns_df = validate_reactions.main()
            self.num_val_rxns += len(val_rxns_df)
            self.num_val_conf_rxns += len(val_rxns_df[val_rxns_df['conf_scores'] > 0.95])            
            
            print(f'Total of Validated ({self.num_val_rxns}) and confident >0.95 ({self.num_val_conf_rxns}) reactions out of the target of {self.frequency_target} for retro_reac: {self.retro_reac} and retro_template: {self.retro_template}')
            
            # append saved reactions until enrichment target is reached
            self.create_folder_if_not_exists(self.results_csv_folder_path)
            self.append_df_to_csv(self.results_csv_path, val_rxns_df)

            # update counters
            self.counter +=1

            # update checkpoint file
            self.update_checkpoint()
        
        print(f'Enrichment finished with counter {self.counter}/999, initial {self.initial_template_frequency} reactions were enriched to {self.num_val_conf_rxns} for retro_reac: {self.retro_reac} and retro_template: {self.retro_template}')
        self.results = pd.read_csv(self.results_csv_path)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Equilibrate class with a YAML config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the Equilibrate class with config values
    eq = Equilibrate(
        dataset_name=config['dataset_name'],
        retro_reac=config['retro_reac'],
        retro_template=config['retro_template'],
        template_line=config['template_line'],
        template_frequency=config['template_frequency'],
        frequency_target=config['frequency_target'],
        T2_model_path=config['T2_model_path'],
        T3_model_path=config['T3_model_path'],
        run_from_path=config['folder_path']
    )

    # Call the process method to start
    eq.process()

if __name__ == '__main__':
    main()

            
            
            
            
            
            
            
            
            
            