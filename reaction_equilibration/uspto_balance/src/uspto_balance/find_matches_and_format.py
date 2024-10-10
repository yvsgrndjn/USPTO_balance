import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import chain
from uspto_balance.dataset_class import Dataset
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()

class FindMatchesApplyTemplateFormatReactions:
    def __init__(self, dataset_name, dataset_version, retro_reac, retro_template, template_line, folder_path):
        self.dataset_name             = dataset_name
        self.dataset_version          = dataset_version
        self.retro_reac               = retro_reac
        self.retro_template           = retro_template
        self.template_line            = template_line
        self.folder_path              = folder_path

        # mol retro_reac
        self.retro_reac_mol           = Chem.MolFromSmarts(self.retro_reac)
        # rxn retro_template
        self.retro_template_rxn       = AllChem.ReactionFromSmarts(self.retro_template)
        # datasets paths
        self.dataset_path             = f'{self.folder_path}data/{self.dataset_name}_{self.dataset_version}.txt'
        self.dataset_mol_path         = f'{self.folder_path}data/{self.dataset_name}_{self.dataset_version}_mol.pkl'



    def load_data_into_df(self):
        
        dataset     = Dataset(file_path=self.dataset_path, file_type='txt')
        dataset_mol = Dataset(file_path=self.dataset_mol_path, file_type='pkl')
        data = {'smiles': dataset.data, 
                'mol': dataset_mol.data}
        df = pd.DataFrame(data)
        # Remove rows with None values in the mol column
        return df[[not el for el in df['mol'].isnull().values]]
        

    def get_rows_with_match_substructures(self):
        match_ind = [i for i, mol in enumerate(self.data_df['mol']) if mol.HasSubstructMatch(self.retro_reac_mol)]
        return self.data_df.iloc[match_ind]
    

    def apply_rxn_templates_on_mols_list(self, mol_list:list):
        return [self.retro_template_rxn.RunReactants([mol]) for mol in mol_list]


    def canonicalize(self, smiles: str) -> str:
        '''
        Converts a smile string into a rdkit canonicalized smile string

        --Inputs--
        smiles (str):   smile string to canonicalize

        --Returns--
        (str) canonicalized smile string
        '''
        return singlestepretrosynthesis.canonicalize_smiles(smiles)


    def format_reaction(self, reactants_tuple: tuple, smi : str) -> list:
        '''Formats canonical reactions as reactants>>product in a smiles format where reactants come from reactants_tuple and the product is smi.

        --Inputs--
        reactants_tuple (tuple(Chem.Mol)):    tuple of reactants Chem.Mol objects resulting from applying a reaction template on a molecule Chem.Mol object
        smi (str):                            product smiles string

        --Returns--
        (list) list of canonicalized reactions in a smiles format
        '''
        # Create a list that will contain the different possible reactions for the given product
        reactants_smiles_list = []

        # Iterate over the different possible reactants combinations for the given product
        for i in range(len(reactants_tuple)):
            reactants_mol = list(reactants_tuple[i])
            reactants_smiles = ''

            # Iterate over the several reactants for a single template application
            for j in range(len(reactants_mol)):

                # Append the reactants in a converted smiles format to the reactants_smiles string
                reactants_smiles += Chem.MolToSmiles(reactants_mol[j]) + '.'

            # Remove the last dot and append to the list containing all the different possible reactions
            reactants_smiles = reactants_smiles[:-1]
            reactants_smiles_list.append(reactants_smiles)
        
        # Remove duplicates
        reactants_smiles_list = list(set(reactants_smiles_list))

        # Canonicalize the reactants smiles
        rxn = [self.canonicalize(reactants_smiles_list[i]) + '>>' + smi for i in range(len(reactants_smiles_list))]
        return rxn


    def remove_incomplete_rxns(self, rxns_list:list):
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


    def main(self):

        self.data_df = self.load_data_into_df()
        self.data_df = self.get_rows_with_match_substructures() # we do not canonicalize again here

        if self.data_df.__len__()==0:
            return
        
        # apply rxn templates on mols list of products to find their corresponding reactants
        corresponding_reactants_list = self.apply_rxn_templates_on_mols_list(self.data_df['mol'].tolist())

        # find indices of empty elements and remove them from both lists ######PUT INTO A FUNCTION######
        ind_remove = [result == () for result in corresponding_reactants_list]
        self.data_df = self.data_df[[not el for el in ind_remove]]
        self.data_df.reset_index(drop=True, inplace=True)
        corresponding_reactants_list = [corresponding_reactants_list[i] for i in range(len(corresponding_reactants_list)) if not ind_remove[i]]
        
        # create fictive reactions in a smiles format ######PUT INTO A FUNCTION######
        fictive_rxns_list = [self.format_reaction(corresponding_reactants_list[k], self.data_df['smiles'][k]) for k in range(self.data_df.__len__())]
        fictive_rxns_list = list(chain.from_iterable(fictive_rxns_list))
        
        # remove empty reactions ######PUT INTO A FUNCTION######
        try:
            fictive_rxns_list.remove('>>') # Remove empty reactions
        except ValueError:
            pass

        fictive_rxns_list = self.remove_incomplete_rxns(fictive_rxns_list)

        return fictive_rxns_list


            
    
    