import re
import os
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from ttlretro.rxnmarkcenter import RXNMarkCenter
rxnmarkcenter = RXNMarkCenter()
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()


def tokenize(reaction:str)->str:
    return singlestepretrosynthesis.smi_tokenizer(reaction)


def find_indices(target_list, target_element):
    return [i for i, el in enumerate(target_list) if el == target_element]


def yield_elements(target_list, indices):
    for i in indices:
        yield target_list[i]


def get_reagents_in_mapped_reaction_AB_C(reaction: str)->list:
    '''
    Takes a mapped reaction A.B>>C string and returns the list
    of reagents (the non-mapped reactants) in the reaction.

    Input:  mapped reaction (str)
    Output: list of reagents (list)
    '''
    regex = re.compile('\[.*:[1-9]*\]')
    strlist = reaction.split('>>')[0].split('.')
    reagents = [string for string in strlist if not re.findall(regex, string)]
    return reagents

def get_reagents_in_reaction_A_B_C(reaction: str)->list:
    '''
    Takes a mapped reaction A>B>C string and returns the list
    of reagents (the non-mapped reactants) in the reaction.

    Input:  mapped reaction (str)
    Output: list of reagents (list)
    '''
    strlist = reaction.split('>')[1].split('.')
    reagents = [string for string in strlist]
    return reagents

def join_reagents(list_reagents:list)->str:
    '''
    Takes a list of reagents and joins them into a single dot-separated
    string.

    Input:  list of reagents (list)
    Output: joined reagents (str)

    example: ['CCO', '[Fe]'] -> 'CCO.[Fe]'
    '''
    return '.'.join(list_reagents)


def get_reactants_in_reaction_AB_C(reaction: str)->list:
    '''
    Takes a mapped reaction im the format A.B>>C (including the unmapped reagents B) string and returns the list
    of reactants (the mapped reactants) in the reaction.

    Input:  mapped reaction (str)
    Output: list of reactants (list)
    '''
    regex = re.compile('\[.*:[1-9]*\]')
    strlist = reaction.split('>>')[0].split('.')
    reactants = [string for string in strlist if re.findall(regex, string)]
    return reactants


def get_reactants_in_reaction_A_B_C(reaction: str)->list:
    '''
    Takes a mapped reaction im the format A>B>C (including the unmapped reagents B) string and returns the list
    of reactants (the mapped reactants) in the reaction.

    Input:  mapped reaction (str)
    Output: list of reactants (list)
    '''
    strlist = reaction.split('>')[0].split('.')
    reactants = [string for string in strlist]
    return reactants


def get_reagents_list_from_mapped_reaction_list_AB_C(reactionlist:list)->list:
    '''
    Takes a list of mapped reactions A.B>>C and returns the associated, same length
    list of reagents

    Input:  list of mapped reactions (list)
    Output: list of reagents (list)
    '''
    reagents_list = []
    for reaction in reactionlist:
        reagents = get_reagents_in_mapped_reaction_AB_C(reaction)
        reagents_list.append(join_reagents(reagents))
    return reagents_list

def get_reagents_list_from_reaction_list_A_B_C(reactionlist:list)->list:
    '''
    Takes a list of (mapped or not) reactions A>B>C and returns the associated, same length
    list of reagents

    Input:  list of mapped reactions (list)
    Output: list of reagents (list)
    '''
    reagents_list = []
    for reaction in reactionlist:
        reagents = get_reagents_in_reaction_A_B_C(reaction)
        reagents_list.append(join_reagents(reagents))
    return reagents_list

def remove_reagents_from_reaction_AB_C(reaction: str)->str:
    '''
    Takes a mapped reaction A.B>>C (which includes the unmapped reagents B) string and removes the reagents to leave only the reactants
    on the left part of the reaction

    Input:  mapped reaction (str)
    Output: mapped reaction without reagents (str)
    '''
    reactants = get_reactants_in_reaction_AB_C(reaction)
    left_side = join_reagents(reactants)
    return left_side + '>>' + reaction.split('>>')[1]


def remove_reagents_from_reaction_list_AB_C(reactionlist:list)->list:
    '''
    Takes a list of mapped reactions A.B>>C (which include the unmapped reagents B) and removes the reagents from them
    
    Input:  list of mapped reactions (list)
    Output: list of mapped reactions without reagents (list)
    '''
    return [remove_reagents_from_reaction_AB_C(reaction) for reaction in reactionlist]    


def format_mapped_reactions_AB_C(df_rxns):
    '''
    Gets the mapped reactions in the format A.B>>C (which contain the unmapped reagents B)from a dataframe at column 'MAPPED_SMILES'
    and returns the list of mapped reactions in the format A>>C (without the reagents B)
    '''
    reactions_AC = remove_reagents_from_reaction_list_AB_C(df_rxns['MAPPED_SMILES'].tolist())
    return reactions_AC


def remove_reagents_from_reaction_A_B_C(reaction: str)->str:
    '''
    Takes a (mapped or not) reaction A>B>C  string and removes the reagents to leave only the reactants
    on the left part of the reaction

    Input:  (mapped or not) reaction (str)
    Output: (mapped or not) reaction without reagents (str)
    '''
    reactants = get_reactants_in_reaction_A_B_C(reaction)
    left_side = join_reagents(reactants)
    return left_side + '>>' + reaction.split('>')[2]


def remove_reagents_from_reaction_list_A_B_C(reactionlist:list)->list:
    '''
    Takes a list of mapped reactions A>B>C and removes the reagents from them
    
    Input:  list of mapped reactions (list)
    Output: list of mapped reactions without reagents (list)
    '''
    return [remove_reagents_from_reaction_A_B_C(reaction) for reaction in reactionlist]    

def tag_products(reaction):
    try:
        return rxnmarkcenter.TagMappedReactionCenter(MappedReaction=reaction, alternative_marking=True, tag_reactants=False)
    except:
        return ''


def tag_reactants(reaction):
    try:
        return rxnmarkcenter.TagMappedReactionCenter(MappedReaction=reaction, alternative_marking=True, tag_reactants=True)
    except:
        return ''


def format_alt_tag_reactions(row):
    try:
        return row[1]['alt_tag_reactants'].split('>>')[0] + '>' + row[1]['reagents'] + '>' + row[1]['alt_tag_products'].split('>>')[1]
    except:
        return '' 


def create_alt_tag_reactions_column(df_rxns: pd.DataFrame)->pd.DataFrame:
    '''
    Takes a dataframe with mapped reactions in a 'MAPPED_SMILES' column and a reagents column called 'reagents'
    and creates a column with the alternative tagged reactions 'alt_tag_reactions' in the format A!>B>C!
    
    -Inputs-
    df_rxns (pd.DataFrame): dataframe with mapped reactions 'MAPPED_SMILES' (with or withoutand reagents columns
    '''
    reactions_AC = format_mapped_reactions(df_rxns)
    processes = os.cpu_count() - 2

    with Pool(processes) as p:
        rxns_AC_tagprod = list(tqdm(p.imap(tag_products, reactions_AC), total=len(reactions_AC)))
        rxns_AC_tagreac = list(tqdm(p.imap(tag_reactants, reactions_AC), total=len(reactions_AC)))

    df_rxns['alt_tag_products']   = rxns_AC_tagprod
    df_rxns['alt_tag_reactants']  = rxns_AC_tagreac

    with Pool(processes) as p:
        alt_tag_reactions = list(tqdm(p.imap(format_alt_tag_reactions, df_rxns.iterrows()), total=len(df_rxns)))

    df_rxns['alt_tag_reactions'] = alt_tag_reactions

    del df_rxns['alt_tag_products']
    del df_rxns['alt_tag_reactants']

    return df_rxns


def get_alt_tag_reactions_from_mapped_reactions_list(reactionList: list) -> list:
    processes = os.cpu_count() - 2
    print('in get_alt_tag_reactions_from_mapped_reactions_list')
    # Use multiprocessing pool to parallelize the operations
    with Pool(processes) as p:
        print('Tagging products...')
        rxns_AC_tagprod = list(tqdm(p.imap(tag_products, reactionList), total=len(reactionList)))
        print('Tagging reactants...')
        rxns_AC_tagreac = list(tqdm(p.imap(tag_reactants, reactionList), total=len(reactionList)))

    alt_tag_reactions = []

    # Error handling for splitting and ensuring we have both parts
    for rxns_AC_tagreac, rxns_AC_tagprod in zip(rxns_AC_tagreac, rxns_AC_tagprod):
        try:
            reac_split = rxns_AC_tagreac.split('>>')
            prod_split = rxns_AC_tagprod.split('>>')
            # Ensure both reaction and product splits have at least 2 parts
            if len(reac_split) < 2 or len(prod_split) < 2:
                raise ValueError("Reaction or product does not contain '>>' or has insufficient parts.")

            # Combine the first part of the reactant with the second part of the product
            alt_tag_reaction = reac_split[0] + '>>' + prod_split[1]
            alt_tag_reactions.append(alt_tag_reaction)

        except (IndexError, ValueError) as e:
            # Handle the error, log or print which reaction caused it for debugging
            print(f"Error processing reaction: reac={rxns_AC_tagreac}, prod={rxns_AC_tagprod}. Error: {e}")
            # You can choose to append an error string or simply continue without appending
            alt_tag_reactions.append('ERROR')  # Or just `continue` if you want to skip

    return alt_tag_reactions


def remove_tagging(reaction):
    return reaction.replace('!','')


def remove_elements_at_indices(lists, error_indices):
    """
    Remove elements from each list in 'lists' at the specified 'error_indices'.

    :param lists: List of lists from which elements need to be removed.
    :param error_indices: Indices of elements to remove.
    :return: List of lists with elements removed.
    """
    # Use set for faster lookups
    error_set = set(error_indices)

    # Remove elements at error indices from each list
    return [[el for idx, el in enumerate(lst) if idx not in error_set] for lst in lists]