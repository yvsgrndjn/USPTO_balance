# class to format the mapped reactions as needed for the template extraction

from rdkit import Chem
import pandas as pd

class ReactionFormatterForTemplateExtraction:
    def __init__(self, dataset):
        """Initialize with a dataset containing mapped reactions."""
        self.dataset = dataset
        self.mapped_rxns_list = self.dataset['MAPPED_SMILES'].tolist()
        self.unmapped_products = []  # Store unmapped products here

    def remove_mapping(self, mol):
        """Remove atom mapping from a molecule."""
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    def get_unmapped_product_smi_from_mapped_reaction_smi(self, mapped_reaction):
        """Process a single mapped reaction to get the unmapped product."""
        mapped_prod_smi = mapped_reaction.split('>>')[1]
        mapped_prod_mol = Chem.MolFromSmiles(mapped_prod_smi)
        unmapped_prod_mol = self.remove_mapping(mapped_prod_mol)
        unmapped_prod_smi = Chem.MolToSmiles(unmapped_prod_mol)
        # Store the unmapped product in the instance variable
        self.unmapped_products.append(unmapped_prod_smi)

    def format_reaction_product_output(self, mapped_reaction):
        """Format the output for the reaction and its unmapped product."""
        # Retrieve the last unmapped product stored
        unmapped_prod_smi = self.unmapped_products[-1]
        return [mapped_reaction, unmapped_prod_smi]

    def format_all_reactions(self):
        """Process all reactions in the dataset and return the results."""
        results = []
        for mapped_reaction in self.mapped_rxns_list:
            # First, get the unmapped product SMILES
            self.get_unmapped_product_smi_from_mapped_reaction_smi(mapped_reaction)
            # Then, format the output for this reaction
            results.append(self.format_reaction_product_output(mapped_reaction))
        return results
