import os
import pandas as pd
from src.uspto_balance.data_handler import DataHandler
import src.uspto_balance.balancing_workflow as bw

class BuildDataset:
    def __init__(self, 
                 path_to_folder, 
                 dataset_name, 
                 df_temp_path, 
                 path_to_save_full_csv, 
                 path_to_ref_dataset, 
                 target_rxns_per_template, 
                 path_to_save_subset_split, 
                 train_size, 
                 skip_appending_rxns_part=False):
        
        self.path_to_folder             = path_to_folder
        self.dataset_name               = dataset_name
        self.df_temp_path               = df_temp_path
        self.path_to_save_full_csv      = path_to_save_full_csv
        self.path_to_ref_dataset        = path_to_ref_dataset
        self.target_rxns_per_template   = target_rxns_per_template
        self.train_size                 = train_size
        self.path_to_save_subset_split  = path_to_save_subset_split

        # initialize DataHandler
        datahandler = DataHandler(file_path=self.df_temp_path)
        self.df_temp = datahandler.load_data()

        if skip_appending_rxns_part:
            datahandler = DataHandler(file_path=self.path_to_save_full_csv)
            self.df = datahandler.load_data()

    def append_rxns_to_csv(self):
        _ = bw.append_saved_rxns_into_csv_old(self.path_to_folder, self.dataset_name, self.df_temp, self.path_to_save_full_csv)

    def load_conf_rxns_df(self):
        # initialize datahandler class
        datahandler = DataHandler(file_path=self.path_to_save_full_csv)
        
        # load the full dataframe
        self.df = datahandler.load_data()
        return self.df[self.df['conf_scores']>0.95].reset_index(drop=True)

    def drop_rxns_in_common_from_df(self):
        # create a column 'isin_original_set' in the dataframe with boolean values
        print('self.df line 48:', self.df)
        print('self.df_ref_data line 49:', self.df_ref_data)
        self.df = bw.label_reactions_from_original_dataframe(self.df, original_rxns=self.df_ref_data['Reaction'], reaction_col='rxns')
        print('self.df line 51:', self.df)
        # drop the reactions that are in the original dataset
        self.df = self.df[self.df['isin_original_set']==False].reset_index(drop=True)
        print('self.df line 54:', self.df)
        # drop the column 'isin_original_set'
        #df = self.df.drop(columns=['isin_original_set'], inplace=True)
        return self.df.drop(columns=['isin_original_set'], inplace=False)
        #print('print df line 57:', df)        
        #return df
    
    def create_dict_of_trainorvalid_templates(self, df, ref_data):
        # get a dictionary of all the templates that are in 'TRAIN' and 'VALID' sets
        print('Getting all the templates in the reference dataset that are in the TRAIN and VALID sets...')
        print('ref_data line 63: ', ref_data)
        print('df line 60:', df)
        trainorvalid_uspto_df = ref_data[ref_data['Set'].isin(['TRAIN', 'VALID'])]
        trainorvalid_templates = df[df['retro_template'].isin(trainorvalid_uspto_df['retro_template_r1_full_corr'])]
        trainorvalid_template_dict = {template: True for template in trainorvalid_templates['retro_template'].unique()}
        return trainorvalid_template_dict
    
    def assign_test_trainvalid_sets(self, trainorvalid_template_dict):
            # assign TEST split to all the reactions
            self.df['Set'] = 'TEST'
            # assign TRAIN/VALID split to all the reactions that are from a template in trainorvalid_template_dict
            self.df.loc[self.df['retro_template'].isin(trainorvalid_template_dict.keys()), 'Set'] = 'TRAIN/VALID'

            # attribute to test set reactions that share a product with a test set from reference dataset
            testset_rxns = self.df_ref_data['Reaction'].loc[self.df_ref_data['Set'] == 'TEST']
            self.df = bw.label_reactions_from_testset(self.df, testset_rxns, 'rxns')
            
            self.testset_df = self.df[self.df['Set']=='TEST']
            self.trainvalidset_df = self.df[self.df['Set']=='TRAIN/VALID']

            return self.testset_df, self.trainvalidset_df


    def get_dataset(self):
        # save all csv files into a single one containing all the valid reactions
        print('Appending all saved reactions into a common csv file...')
        self.append_rxns_to_csv() # to be optimized in balancing_workflow, appending directly and saving only once, not having to reload it afterwards
        print('Done and saved to:', self.path_to_save_full_csv)

        # load the dataframe with only the confident reactions
        self.df = self.load_conf_rxns_df()

        # add a products column to the dataframe
        print('Adding a products column to the dataframe...')
        self.df = bw.add_products_column(self.df) # to be optimized in balancing_workflow

        self.df.to_csv(self.path_to_save_full_csv)
        return self.df
    
    def split_dataset(self):
        # if a reference dataset exists, load it and use it to assign the splits to the newly create dataset?
        if os.path.exists(self.path_to_ref_dataset):
            # load the reference dataset
            datahandler_refdataset = DataHandler(file_path=self.path_to_ref_dataset)
            print('Loading reference dataset...')
            self.df_ref_data = datahandler_refdataset.load_data()
            
            # remove the reactions in the created dataset already in the reference dataset
            # -- create a column 'isin_original_set' in the dataframe with boolean values
            print('Removing reactions that are also present in the original dataset...')
            print('self.df line 107: ',self.df)
            self.df = self.drop_rxns_in_common_from_df()
            print('self.df line 109: ',self.df)
            # get a dictionary of all the templates that are in 'TRAIN' and 'VALID' sets
            trainorvalid_template_dict = self.create_dict_of_trainorvalid_templates(df=self.df, ref_data=self.df_ref_data)

            # define 'TEST' and 'TRAIN/VALID' sets
            self.testset_df, self.trainvalidset_df = self.assign_test_trainvalid_sets(trainorvalid_template_dict= trainorvalid_template_dict)

            # take self.target_rxns_per_template reactions per template in the train/validation set
            print(f'Subsetting {self.target_rxns_per_template} reactions per template in the train/validation set...')
            self.trainvalidset_df = bw.subset_n_random_reactions_per_template(self.trainvalidset_df, target=self.target_rxns_per_template)

            print('Splitting the train/validation set into TRAIN and VALID sets...')
            self.trainvalidset_df = self.trainvalidset_df.groupby('retro_template', group_keys=False).apply(lambda x: bw.stratified_split(x, self.train_size))

            # concat the full test set and the train/valid subsets
            print('Concatenating the test set and the train/validation set...')
            self.df = pd.concat([self.trainvalidset_df, self.testset_df])

            print('Saving the final dataset...')
            self.df.to_csv(self.path_to_save_subset_split)

        # if there is no reference dataset, split the dataset into 'TRAIN', 'TEST', and 'VAL' sets according to templates
        else:
            self.df = self.df.groupby('retro_template', group_keys=False).apply(lambda x: bw.stratified_train_val_test_split(x, self.train_size))
            self.df.to_csv(self.path_to_save_subset_split)

