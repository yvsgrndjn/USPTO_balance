import os
import pandas as pd
from tqdm import tqdm
import src.uspto_balance.retrosynthesis_utilities as retrosynthesis_utilities


class CreateSrcTgtFiles:
    def __init__(self, mapped_smiles, splits, temp_list, dataset_name, base_path, target_rxns_per_template):
        self.mapped_smiles                  = mapped_smiles
        self.splits                         = splits
        self.temp_list                      = temp_list
        self.dataset_name                   = dataset_name
        self.base_path                      = base_path
        self.target_reactions_per_template  = target_rxns_per_template
        self.temp_series = pd.Series()
        
        # Process the SMILES and tag reactions
        print('Obtaining reagents from mapped reactions...')
        self.reagents = retrosynthesis_utilities.get_reagents_list_from_reaction_list_A_B_C(self.mapped_smiles)

        print('Obtaining reactions without reagents...')
        self.reactions_AC = retrosynthesis_utilities.remove_reagents_from_reaction_list_A_B_C(self.mapped_smiles)

        print('Obtaining tagged reactions...')
        self.tagged_reaction_AC = retrosynthesis_utilities.get_alt_tag_reactions_from_mapped_reactions_list(self.reactions_AC)

    def remove_errors_from_lists(self):
        lists_to_clean = [self.mapped_smiles, 
                          self.splits, 
                          self.reagents, 
                          self.tagged_reaction_AC, 
                          self.temp_list, 
                          ]
        iserror = [el == 'ERROR' for el in self.tagged_reaction_AC]
        error_indices = [el for el in range(len(self.tagged_reaction_AC)) if iserror[el]]
        cleaned_lists = retrosynthesis_utilities.remove_elements_at_indices(lists_to_clean, error_indices)
        self.mapped_smiles, self.splits, self.reagents, self.tagged_reaction_AC, self.temp_list = cleaned_lists

    def get_train_test_valid_indices(self, data_set):
        """Find indices for 'TRAIN', 'TEST', or 'VALID' splits."""
        return retrosynthesis_utilities.find_indices(self.splits, data_set)

    def generate_datasets(self, indices):
        """Generate source and target datasets based on the indices."""
        data_sources = {
            'T1_src': self.tok_tagged_C,
            'T1_tgt': self.tok_A,
            'T2_src': self.tok_reaction_AC,
            'T2_tgt': self.tok_reagents,
            'T3_src': None,  # T1_tgt assigned later
            'T3_tgt': self.tok_C,
            'T3FT_src': self.tok_tagged_A,
            'T3FT_tgt': None,  # T3_tgt assigned later
        }

        result = {}
        for key, source in data_sources.items():
            if source is not None:
                result[key] = list(retrosynthesis_utilities.yield_elements(source, indices))

        # Assign 'T3' and 'T3FT' specific cases
        result['T3_src'] = result['T1_tgt']
        result['T3FT_tgt'] = result['T3_tgt']
        return result

    def save_files(self, result, dataset_type):
        """Save the generated datasets to files."""
        for key, data_list in result.items():
            var_name = f"{key}_{dataset_type}"
            self.save_list_to_file(var_name, data_list)

    def save_list_to_file(self, var_name, data_list):
        """Helper method to save a list to a file."""
        parts = var_name.split('_')
        folder_name = parts[0]
        file_name = '_'.join(parts[1:]) + '.txt'

        # Create directory if it doesn't exist
        dir_path = os.path.join(self.base_path,'data/models', folder_name, self.dataset_name, self.target_reactions_per_template+'_rxns_per_template')
        os.makedirs(dir_path, exist_ok=True)

        # Save the list to the file
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    def process_and_save_all(self):
        """Process and save all datasets for TRAIN, TEST, and VALID."""
        print('Removing elements with A>>C mapping error from all lists...')
        self.remove_errors_from_lists()

        print('Obtaining non tagged A>>C...')
        self.reaction_AC = list(map(retrosynthesis_utilities.remove_tagging, self.tagged_reaction_AC)) # Would it be simpler to obtain from self.mapped_smiles?
        
        print('Tokenizing non tagged A>>C...')
        self.tok_reaction_AC = list(map(retrosynthesis_utilities.tokenize, self.reaction_AC))
        # Generate tok_A and tok_C

        print('Obtaining non-tagged A and non-tagged C...')
        self.tok_A, self.tok_C = zip(*[el.split(' > > ') for el in self.tok_reaction_AC])

        print('Tokenizing tagged A and tagged C...')
        self.tok_tagged_A, self.tok_tagged_C = zip(*[(retrosynthesis_utilities.tokenize(el.split('>>')[0]), retrosynthesis_utilities.tokenize(el.split('>>')[1])) for el in self.tagged_reaction_AC])
        
        print('Tokenizing reagents...')
        self.tok_reagents = list(map(retrosynthesis_utilities.tokenize, self.reagents))

        self.temp_series = pd.Series(self.temp_list)

        for dataset_type in ['TRAIN', 'TEST', 'VALID']:

            print('Obtaining splits indices')
            indices = self.get_train_test_valid_indices(dataset_type)

            print('Generating datasets...')
            result = self.generate_datasets(indices)

            print('Saving files...')
            self.save_files(result, dataset_type.lower())

            # added after to get template lists for the different splits, could be cleaner
            if self.temp_list =='':
                print('In CreateSrcTgtFiles.process_and_save_all(): No template column name provided. No template files will be saved.') 
            else:
                temp_series_split = self.temp_series.iloc[indices]
                file_path = os.path.join(
                    self.base_path,
                    'data/models',
                    'templates',
                    self.dataset_name,
                    f"{self.target_reactions_per_template}_rxns_per_template/templates_{dataset_type}.txt",
                    )
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))
                
                temp_series_split.to_csv(file_path, header=False)

#import pandas as pd
#
#from ttlretro.single_step_retro import SingleStepRetrosynthesis
#singlestepretrosynthesis = SingleStepRetrosynthesis()
#
#from singlestepretrosynthesis import smi_tokenizer
#from uspto_balance import balancing_workflow as bw
#
#class CreateSrcTgtFiles:
#
#    def __init__(self, df:pd.DataFrame):
#        self.df = df
#    
#    @staticmethod
#    def remove_tags(reaction):
#        return reaction.replace('!', '')
#
#    @staticmethod
#    def tokenize(reaction):
#        return smi_tokenizer(reaction)
#
#    def process_reaction(self, reaction, token_index):
#        parts = reaction.split('>')
#        if token_index == 0:
#            return self.tokenize(self.remove_tags(parts[0]))
#        elif token_index == 1:
#            return self.tokenize(parts[1])
#        elif token_index == 2:
#            return self.tokenize(parts[2])
#
#    def process_reaction_list(self, reaction_list, token_index):
#        return [self.process_reaction(reaction, token_index) for reaction in reaction_list]
#
#    def find_split_indices(self):
#        train_ind = self.df[self.df['Set'] == 'TRAIN'].index.tolist()
#        test_ind = self.df[self.df['Set'] == 'TEST'].index.tolist()
#        val_ind = self.df[self.df['Set'] == 'VAL'].index.tolist()
#        return train_ind, test_ind, val_ind
#
#    def split_list(self, reaction_list, train_ind, test_ind, val_ind):
#        return [reaction_list[i] for i in train_ind], [reaction_list[i] for i in test_ind], [reaction_list[i] for i in val_ind]
#
#    def save_data(self, path, data, prefix, split):
#        filename_src = f'src_{split}.txt'
#        filename_tgt = f'tgt_{split}.txt'
#        bw.save_list_to_txt(path, data[0], f'{prefix}_{filename_src}')
#        bw.save_list_to_txt(path, data[1], f'{prefix}_{filename_tgt}')
#
#    def process_and_save(self, path, token_index, prefix):
#        reaction_list = self.df['alt_tag_reactions'].tolist()
#        train_ind, test_ind, val_ind = self.find_split_indices()
#        reaction_train, reaction_test, reaction_val = self.split_list(reaction_list, train_ind, test_ind, val_ind)
#        data_train = self.process_reaction_list(reaction_train, token_index)
#        data_test = self.process_reaction_list(reaction_test, token_index)
#        data_val = self.process_reaction_list(reaction_val, token_index)
#        self.save_data(path, (data_train, data_test), prefix, 'train')
#        self.save_data(path, (data_test, data_test), prefix, 'test')
#        self.save_data(path, (data_val, data_val), prefix, 'val')    
#
# Usage:
#uspto_df_r0r1 = pd.DataFrame()  # Assuming this is defined elsewhere
#processor = ReactionProcessor(uspto_df_r0r1)

# Process and save for src T1
#processor.process_and_save('/home/yves/Documents/GitHub/test_folder_uspto_balance/models/USPTO_r0r1/T1/', 0, 'src')

# Process and save for tgt T1
#processor.process_and_save('/home/yves/Documents/GitHub/test_folder_uspto_balance/models/USPTO_r0r1/T1/', 2, 'tgt')
