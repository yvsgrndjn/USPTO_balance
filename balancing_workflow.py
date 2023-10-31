import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import pickle
from itertools import chain
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()

# Functions related to templates handling
def smiles_to_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol


def load_subsets(retro_reac, GDB_version: str = '', template_version: str = ''): 

    name = f'GDB13S_sub_{retro_reac}'
    folder_path = f'./GDB_subsets_{GDB_version}_{template_version}'
    folder_path_mol = f'./GDB_subsets_mol_{GDB_version}_{template_version}'

    try:
        with open(f'{folder_path}/{name}.txt', 'r') as f:
            GDB13S_sub = [line.strip() for line in f]
        with open(f'{folder_path_mol}/{name}.pkl', 'rb') as f:
            GDB13S_sub_mol = pickle.load(f)

        return GDB13S_sub, GDB13S_sub_mol

    except FileNotFoundError:
        print(f'No subsets found for retro_reac: {retro_reac}')
        return [], []


def load_template_version(template_version):
    df_templates_split = pd.read_pickle(f'/home/yves/Documents/GitHub/USPTO_balance/data/templates_split/df_templates_to_enrich_part_{template_version}.pkl')
    return df_templates_split


#def load_rxns(GDB_version, template_version, retro_reac, retro_template):
#    '''
#    Loads the rxns list that were created in part 2 of the GDB subset matching the retro_reac pattern and applying retro_template to it
#    '''
#    try:
#        with open(f'/home/yves/Documents/GitHub/USPTO_balance/created_rxns_{GDB_version}_{template_version}/rxns_{retro_reac}_{retro_template}.txt', 'r') as f:
#            rxns_list = []
#            for line in f:
#                rxns_list.append(line.split('\n')[0])
#        return rxns_list
#    
#    except FileNotFoundError:
#        print(f'No reactions found for retro_reac: {retro_reac} and retro_template: {retro_template}')
#        return []

def load_rxns(GDB_version, template_version, retro_reac, retro_template, rxns_number = 10000):
    '''
    Loads the rxns list that were created in part 2 of the GDB subset matching the retro_reac pattern and applying retro_template to it
    '''
    try:
        with open(f'/home/yves/Documents/GitHub/USPTO_balance/created_rxns_{GDB_version}_{template_version}/rxns_{retro_reac}_{retro_template}.txt', 'r') as f:
            rxns_list = []
            rxns_list = list(islice(f, rxns_number))
            rxns_list = [rxns_list[i].split('\n')[0] for i in range(len(rxns_list))]
        return rxns_list
    
    except FileNotFoundError:
        print(f'No reactions found for retro_reac: {retro_reac} and retro_template: {retro_template}')
        return []

def tokenize_rxn_list(rxns_list):
    '''
    Tokenizes a list of reactions
    '''
    tok_rxns_list = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_list]
    return tok_rxns_list

def canonicalize(smiles):
    '''
    Converts a smile string into a rdkit canonicalized smile string
    '''
    return singlestepretrosynthesis.canonicalize_smiles(smiles)


def save_rxns(rxns_list, retro_reac, retro_template, GDB_version: str = '', template_version: str = ''):
    if rxns_list:
        name = f'rxns_{retro_reac}_{retro_template}'
        folder_path = f'./created_rxns_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in rxns_list:
                f.write(item + '\n')
        print(f'Saved {len(rxns_list)} reactions for retro_reac: {retro_reac} and retro_template: {retro_template} and folder : {GDB_version}_{template_version}')



def convert_and_save_subset(subset, subset_mol, retro_reac, GDB_version: str = '', template_version: str = ''):
    '''
    Saves a subset of SMILES strings to a txt file and converts it to mol before saving it to a pkl file
    '''
    if subset:
        name = f'GDB13S_sub_{retro_reac}'
        folder_path = f'./GDB_subsets_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
        with open(f'{folder_path}/{name}.txt', 'w') as f:
            for item in subset:
                f.write(item + '\n')

        folder_path_mol = f'./GDB_subsets_mol_{GDB_version}_{template_version}'

        if not os.path.exists(folder_path_mol):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path_mol)
        with open(f'{folder_path_mol}/{name}.pkl', 'wb') as f:
            pickle.dump(subset_mol, f)

def select_templates_to_enrich(data: pd.Series, min_number: int = 10, target_number: int = 10000):
    """
    Takes a Pandas Series of template hashes and returns the list of hashes that meet enrichment requirements along with their frequencies.

    data:           Pandas Series of template hashes
    min_number:     minimum number of occurrences of a template hash to be considered for enrichment
    target_number:  target number of occurrences of a template hash after enrichment
    """
    # Calculate the number of occurrences for each template hash
    template_counts = data.value_counts()

    # Filter the template counts to select hashes within the specified range
    selected_hashes = template_counts[
        (template_counts >= min_number) & (template_counts <= target_number)
    ]

    # Extract the index (template hashes) as a list
    selected_hashes_list = selected_hashes.index.tolist()

    selected_frequencies_list = selected_hashes.sort_values(ascending=False).tolist()

    return selected_hashes_list, selected_frequencies_list



def rxn_smarts_to_sanitized_reactant_smarts(smarts):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns its reactant side in a rdkit mol format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        smarts = smarts.split('>>')[0]
        if smarts[0]=='(' and smarts[-1] == ')':
            smarts = smarts[1:-1].replace(').(', '.')

        mol = Chem.MolFromSmarts(smarts)
        return mol
    except:
        return ''



def rxn_smarts_to_sanitized_product_smarts(smarts):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns its product side in a rdkit mol format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        smarts = smarts.split('>>')[1]
        if smarts[0]=='(' and smarts[-1] == ')':
            smarts = smarts[1:-1].replace(').(', '.')

        mol = Chem.MolFromSmarts(smarts)
        return mol
    except:
        return ''



def rxn_smarts_to_formatted_rxn_smarts(smarts:str):
    '''
    Takes as input the reaction template string in CASP tempalte extraction format, and returns it in a rdkit acceptable format, returns nothing if the template is not rdkit acceptable
    '''
    try:
        reac = smarts.split('>>')[0]
        prod = smarts.split('>>')[1]
        smarts = smarts.split('>>')[1]
        if smarts[0]=='(' and smarts[-1] == ')':
            reac = reac[1:-1].replace(').(', '.')
            prod = prod[1:-1].replace(').(', '.')
            
        rxnsmarts = AllChem.ReactionFromSmarts(reac + '>>' + prod)
        return rxnsmarts
    except:
        return ''


def process_retro_template(retro_reac, retro_template, GDB_version: str = '', template_version: str = ''):
    
    GDB13S_sub, GDB13S_sub_mol = load_subsets(retro_reac, GDB_version, template_version)

    if not GDB13S_sub:
        return

    # Apply template
    GDB13S_sub_app_temp = apply_rxn_template_on_mols_list(GDB13S_sub_mol, retro_template)

    # Find indices of empty elements and remove them from both lists
    ind_remove = [GDB13S_sub_app_temp[i] == () for i in range(len(GDB13S_sub_app_temp))]
    GDB13S_sub_app_temp_sort = [GDB13S_sub_app_temp[i] for i in range(len(GDB13S_sub_app_temp)) if not ind_remove[i]]
    GDB13S_sub_sort = [GDB13S_sub[i] for i in range(len(GDB13S_sub)) if not ind_remove[i]]

    # Create fictive reactions
    fictive_rxns_list = [format_reaction(GDB13S_sub_app_temp_sort[k], GDB13S_sub_sort[k]) for k in range(len(GDB13S_sub_sort))]
    fictive_rxns_list = list(chain.from_iterable(fictive_rxns_list))

    # Save in a txt file
    save_rxns(fictive_rxns_list, retro_reac, retro_template, GDB_version, template_version)



def find_reaction_template_of_hash(fulltemplate_df:pd.DataFrame, template):
    '''
    For a given template, returns its associated most frequent reaction template (column 'retro_template') from the full template dataframe
    '''
    return fulltemplate_df[fulltemplate_df['template_hash'] == template]['retro_template'].value_counts().keys().tolist()[0]



#def extract_match_smiles_from_dataset_old(dataset:list, dataset_mol:list, template:str):
#    """
#    This function extracts the elements from a smiles dataset that match a certain template and canonicalizes them
#    """
#    template_mol    = Chem.MolFromSmarts(template)
#    match_ind       = [i for i in range(len(dataset_mol)) if dataset_mol[i].HasSubstructMatch(template_mol)]
#    dataset_sub     = [canonicalize(dataset[i]) for i in range(len(dataset)) if i in match_ind]
#    dataset_sub_mol = [dataset_mol[i] for i in range(len(dataset_mol)) if i in match_ind]
#    return dataset_sub, dataset_sub_mol


def extract_match_smiles_from_dataset(dataset:list, dataset_mol:list, template:str):
    """
    This function extracts the elements from a smiles dataset that match a certain template and canonicalizes them
    """
    #convert template to mol
    template_mol    = Chem.MolFromSmarts(template)

    #find indices in the dataset of the substructure matches
    match_ind = [i for i, mol in enumerate(dataset_mol) if mol.HasSubstructMatch(template_mol)]

    #create a subset of canonicalized smiles containing the substructure matches
    dataset_match = [dataset[i] for i in match_ind]
    processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=processes)
    dataset_sub = pool.map(canonicalize, dataset_match)
    pool.close()
    pool.join()

    #create subset of mol objects containing the substructure matches
    dataset_sub_mol = [dataset_mol[i] for i in match_ind]
    return dataset_sub, dataset_sub_mol


def apply_rxn_template_on_mols_list(dataset_mol:list, rxn_template:str):
    '''
    Applies a reaction template on a list of mols (dataset) and returns a list of lists of reactants

    dataset: list of molecules mols 
    rxn_template: reaction template in string format
    ---
    returns: list of lists of reactants
    '''
    rxn = AllChem.ReactionFromSmarts(rxn_template)
    return [rxn.RunReactants((dataset_mol[i],)) for i in range(len(dataset_mol))]


def format_reaction(reactants_tuple, smi):
    '''From the runreactants result, returns the reactions in a smiles format'''

    reactants_smiles_list = []
    for i in range(len(reactants_tuple)):
        reactants_mol = list(reactants_tuple[i])

        reactants_smiles = ''
        #reactants_smiles_list = []
        for j in range(len(reactants_mol)):
            reactants_smiles += Chem.MolToSmiles(reactants_mol[j]) + '.'
        reactants_smiles = reactants_smiles[:-1]
        reactants_smiles_list.append(reactants_smiles)
    
    reactants_smiles_list = list(set(reactants_smiles_list))
    rxn = [reactants_smiles_list[i] + '>>' + smi for i in range(len(reactants_smiles_list))]
    
    return rxn


def run_T2_predictions(tok_rxns_list, Model_path, beam_size: int = 1, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>>product) and predicts the needed reagents for the reaction to take place.
    Gives back the list of reagents that has the same length as the input SMILES list.
    '''
    [preds_T2, probs_T2] = singlestepretrosynthesis.Execute_Prediction(tok_rxns_list, Model_path, beam_size, batch_size, untokenize_output)
    return preds_T2[0]


def prepare_rxns_T2_for_T3(rxns_list, preds_T2):
    '''
    Takes a list of reactions in SMILES format (reactants>>product) and the predicted reagents for each reaction
    Outputs:
        - rxns_T2_list: a list of reactions in the format reactants>reagent>product,
        - rxns_T2_to_T3: a list of tokenized reactions in the format reactants>reagent,
        - rxns_T2_to_T3_tok: a list of tokenized reactions in the format reactants>reagent to use as input for T3
    '''
    rxns_T2_list = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] + '>' + rxns_list[i].split('>>')[1] for i in range(len(preds_T2))]
    rxns_T2_to_T3 = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] for i in range(len(preds_T2))]
    rxns_T2_to_T3_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_T2_to_T3]
    return rxns_T2_list, rxns_T2_to_T3, rxns_T2_to_T3_tok


def run_T3_predictions(rxns_T2_to_T3_tok, Model_path, beam_size: int = 3, batch_size: int = 64, untokenize_output:bool = True):
    '''
    Takes a SMILES list (list of tokenized reactions in the format reactants>reagents) and performs forward prediction on them.
    Gives back the list of predicted products that has the same length as the input SMILES list.    
    '''
    [preds_T3, probs_T3] = singlestepretrosynthesis.Execute_Prediction(rxns_T2_to_T3_tok, Model_path, beam_size, batch_size, untokenize_output)
    return preds_T3[0], probs_T3[0]


def find_ind_match_T3_preds_ref(preds_T3, rxns_list):
    '''
    Takes as input the forward predictions of T3 and the original reactions list, returns the indices of the predicted products matching the original products (returns indices of forward validated reactions)
    '''
    #canonicalize preds_T3
    preds_T3 = [singlestepretrosynthesis.canonicalize_smiles(i) for i in preds_T3]

    preds_ref = [rxns_list[i].split('>>')[1] for i in range(len(rxns_list))]
    ind = [i for i in range(len(preds_T3)) if preds_T3[i] == preds_ref[i]]
    return ind


def keeps_match_confident_rxns(rxns_list, probs_T3, ind_match, conf_score = 0.9):
    '''
    
    '''
    ind_keep = [probs_T3[i] > conf_score for i in range(len(probs_T3))]
    rxns_conf = [rxns_list[i] for i in range(len(ind_keep)) if ind_keep[i] == True and i in ind_match]
    return rxns_conf


def save_conf_rxns(rxns_conf, GDB_version, template_version, retro_reac, retro_template):

    folder_path = f'saved_rxns_{GDB_version}_{template_version}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'/home/yves/Documents/GitHub/USPTO_balance/{folder_path}/rxns_{retro_reac}_{retro_template}.txt', 'w') as f:
        for item in rxns_conf:
            f.write(item + '\n')


def delete_evaluated_rxns(GDB_version, template_version, retro_reac, retro_template):

    name = f'rxns_{retro_reac}_{retro_template}'
    folder_path = f'./created_rxns_{GDB_version}_{template_version}'
    os.remove(f'{folder_path}/{name}.txt')








