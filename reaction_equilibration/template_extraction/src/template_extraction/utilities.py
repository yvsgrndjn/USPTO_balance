# use with rxnutils environment
import pandas as pd
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction
import os
from tqdm import tqdm
from multiprocessing import Pool

#define function for extraction of retro template
def extract_retro_template_and_hashes_with_check(input):
    reaction = input[0]
    product = input[1]
    
    try:
        rxn = ChemicalReaction(reaction)
    except ValueError: 
        return False, 'rxn error', '', '', ''
    
    try:
        rxn.generate_reaction_template(radius=0)
    except:
        return False, 'template error', '', '', ''
        
    try:
        reactant_list = rxn.retro_template.apply(product)
        return True, rxn.retro_template.smarts, rxn.retro_template.hash_from_bits(), rxn.retro_template.hash_from_smarts(), rxn.retro_template.hash_from_smiles()
    except ValueError:
        return False, rxn.retro_template.smarts, '', '', ''
    except RuntimeError as e:
        return False, f'RuntimeError: {e}', '', '', ''