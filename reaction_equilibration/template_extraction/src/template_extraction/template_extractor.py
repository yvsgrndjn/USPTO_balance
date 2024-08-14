# class to extract templates

from rdkit import Chem
import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction


class TemplateExtractor:
    def __init__(self, reaction_product_pairs):
        """Initialize with reaction-product pairs."""
        self.reaction_product_pairs = reaction_product_pairs
        self.radius_0_results = []
        self.radius_1_results = []

    def extract_retro_template_and_hashes(self, reaction, product, radius=0):
        """Extract retro templates and hashes from a reaction and product pair."""
        try:
            rxn = ChemicalReaction(reaction)
        except ValueError:
            return False, 'rxn error', ''

        try:
            rxn.generate_reaction_template(radius=radius)
        except:
            return False, 'template error', ''

        try:
            reactant_list = rxn.retro_template.apply(product)
            return True, rxn.retro_template.smarts, rxn.retro_template.hash_from_smarts()
        except ValueError:
            return False, rxn.retro_template.smarts, ''
        except RuntimeError as e:
            return False, f'RuntimeError: {e}', ''

    def extract_r0_templates_for_all(self):
        for reaction, product in self.reaction_product_pairs:
            # Extract and store results for radius 0
            result_radius_0 = self.extract_retro_template_and_hashes(reaction, product, radius=0)
            self.radius_0_results.append({
                'reaction': reaction,
                'product': product,
                'radius': 0,
                'success': result_radius_0[0],
                'smarts': result_radius_0[1],
                'hash_from_smarts': result_radius_0[2],
            })
        return self.radius_0_results
    
    def extract_r1_templates_for_all(self):
        for reaction, product in self.reaction_product_pairs:
            result_radius_1 = self.extract_retro_template_and_hashes(reaction, product, radius=1)
            self.radius_1_results.append({
                'reaction': reaction,
                'product': product,
                'radius': 1,
                'success': result_radius_1[0],
                'smarts': result_radius_1[1],
                'hash_from_smarts': result_radius_1[2],
            })
        return self.radius_1_results
