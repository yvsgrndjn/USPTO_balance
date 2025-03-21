import os
import pandas as pd
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()



class ValidateReactionsWithTTL:
    def __init__(self, reactions_list: list, T2_model_path: str, T3_model_path: str):
        self.rxns_list            = reactions_list
        self.T2_model_path        = T2_model_path
        self.T3_model_path        = T3_model_path
        

    def tokenize_rxn_list(self, rxns_list: list):
        '''
        Tokenizes a list of reactions

        --Inputs--
        rxns_list (list):             List of reactions in the format reactants>>product

        --Returns--
        tok_rxns_list (list):         List of tokenized reactions in the format r e a c t a n t s >> p r o d u c t
        '''
        tok_rxns_list = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_list]
        return tok_rxns_list

    def ensure_tmp_folder(self):
        if not os.path.exists('./tmp/'):
            os.makedirs('./tmp/')

    def run_T2_predictions(self, tok_rxns_list: list, beam_size: int = 1, batch_size: int = 64, untokenize_output:bool = True):
        '''
        Takes a SMILES list (list of tokenized reactions in the format reactants>>product) and predicts the needed reagents for the reaction to take place.
        Gives back the list of reagents, which is same length as the input SMILES list.
        --Inputs--
        tok_rxns_list (list(str)):   List of tokenized reactions in the format reactants>>product
        Model_path (str):             Path to the model to use for prediction
        beam_size (int):              Beam size to use for prediction (default = 1)
        batch_size (int):             Batch size to use for prediction (default = 64)
        untokenize_output (bool):     Whether to untokenize the output or not (default = True)

        --Returns--
        preds_T2 (list(str)):         List of predicted reagents for each of the input reactions of 'tok_rxns_list'
        '''
        self.ensure_tmp_folder()
        [preds_T2, probs_T2] = singlestepretrosynthesis.Execute_Prediction(tok_rxns_list, self.T2_model_path, beam_size, batch_size, untokenize_output)
        return preds_T2[0]

    def remove_unmapped_rxns(self, MappedReactions: list, preds_T2: list, rxns_list: list):
        '''
        Remove the reactions for which mapping was unsuccessful, characterised by '>>'

        --Inputs--
        MappedReactions (list(str)):    List of mapped reactions in the format reactants(mapped)>>product(mapped)
        preds_T2 (list(str)):           List of predicted reagents for each of the input reactions of 'rxns_list'
        rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped)

        --Returns--
        MappedReactions (list(str)):    List of mapped reactions in the format reactants(mapped)>>product(mapped) without the unmapped reactions
        preds_T2 (list(str)):           List of predicted reagents for each of the input reactions of 'rxns_list' without the predictions corresponding to the unmapped reactions
        rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped) without the reactions corresponding to the unmapped reactions
        '''
        indices_to_remove = [i for i, value in enumerate(MappedReactions) if value == '>>']

        for index in reversed(indices_to_remove):
            del MappedReactions[index]
            del preds_T2[index]
            del rxns_list[index]
        return MappedReactions, preds_T2, rxns_list

    def format_rxns_for_T3(self, rxns_list: list, preds_T2: list):
        '''
        From the rxns_list and the predicted reagents for each reaction, returns a list of tokenized reactions
        in an appropriate format to use as input to forward tag T3
        tok(tag(reactants) > reagents )

        --Inputs--
        rxns_list (list(str)):              List of reactions in the format reactants>>product
        preds_T2 (list(str)):               List of predicted reagents for each reaction in 'rxns_list'

        --Returns-- 
        reconstructed_rxns_tok (list(str)): List of tokenized reactions in the format: reactants(tagged with "!") > reagents (tokenized) ready to be input into forward-tag Forward validation (T3-FT)
        MappedReactions (list(str)):        List of mapped reactions in the format reactants(mapped)>>product(mapped)
        '''
        MappedReactions = list(singlestepretrosynthesis.rxn_mapper_batch.map_reactions(rxns_list))
        MappedReactions, preds_T2, rxns_list = self.remove_unmapped_rxns(MappedReactions, preds_T2, rxns_list)
        taggedreactants = [singlestepretrosynthesis.rxn_mark_center.TagMappedReactionCenter(MappedReactions[i], alternative_marking = True, tag_reactants = True).split('>>')[0] for i in range(len(MappedReactions))]
        rxns_format_T3 = [taggedreactants[i] + '>' + preds_T2[i] for i in range(len(preds_T2))]
        rxns_format_T3_tok = [singlestepretrosynthesis.smi_tokenizer(i) for i in rxns_format_T3]

        return rxns_format_T3_tok, MappedReactions

    def run_T3_predictions(self, rxns_format_T3_tok: list, Model_path: str, beam_size: int = 3, batch_size: int = 64, untokenize_output:bool = True):
        '''
        Takes a SMILES list (list of tokenized reactions in the format reactants>reagents) and performs forward prediction on them.
        Gives back the list of predicted products that has the same length as the input SMILES list.  

        --Inputs--
        rxns_format_T3_tok (list(str)):   List of tokenized reactions in the format: reactants(tagged with "!") >reagents (tokenized)
        Model_path (str):                Path to the model to use for prediction (here forward validation, preferably with forward-tag)
        beam_size (int):                 Beam size to use for prediction (default = 3)
        batch_size (int):                Batch size to use for prediction (default = 64)
        untokenize_output (bool):        Whether to untokenize the output or not (default = True)

        --Returns--
        preds_T3[0] (list(str)):            List of predicted products for each of the input reactions of 'rxns_T2_to_T3_tok'
        probs_T3[0] (list(float)):          List of confidence scores for each of the predicted products of 'preds_T3', in [0,1]
        '''
        [preds_T3, probs_T3] = singlestepretrosynthesis.Execute_Prediction(rxns_format_T3_tok, Model_path, beam_size, batch_size, untokenize_output)
        return preds_T3[0], probs_T3[0]

    def get_match_indices(self, preds_T3: list, rxns_list: list):
        '''
        Performs forward validation on the predictions of T3 and the original reactions list (used as ground truth). Returns the matches between the two lists.
        Takes as input the forward predictions of T3 and the original reactions list, returns the indices of the predicted products matching the original
        products (returns indices of forward validated reactions)

        --Inputs--
        preds_T3 (list(str)):           List of predicted products for each of the input reactions of 'rxns_T2_to_T3_tok' (output from run_T3_predictions)
        rxns_list (list(str)):          List of reactions in the format reactants>>product (not mapped) representing the ground truth

        --Returns--
        ind_match (list(int)):          List of indices of the forward validated reactions
        '''
        # Canonicalization of the predictions
        preds_T3 = [singlestepretrosynthesis.canonicalize_smiles(smiles) for smiles in preds_T3]

        # Compare predictions and ground truth (original products), keep indices of matches
        preds_ref = [singlestepretrosynthesis.canonicalize_smiles(rxn.split('>>')[1]) for rxn in rxns_list] 
        ind_match = [i for i in range(len(preds_T3)) if preds_T3[i] == preds_ref[i]]
        return ind_match    
    
    def add_reagents_to_rxns_list(self, rxns_list, preds_T2, ind_match):
        '''
        Add reagents to the list of reactions 'rxns_list' that are forward validated (at indices 'ind_match') . Final reaction format: reactant(s)>reagent(s)>product.

        --Inputs--
        rxns_list (list(str)):          List of reactions in the format reactants>>product 
        preds_T2 (list(str)):           List of predicted reagents for each reaction in 'rxns_list'
        ind_match (list(int)):          List of indices of the forward validated reactions

        --Returns--
        rxns_list_with_reagents (list(str)): List of validated reactions with format: reactant(s)>reagent(s)>product
        '''
        rxns_list_with_reagents = [rxns_list[i].split('>>')[0] + '>' + preds_T2[i] + '>' + rxns_list[i].split('>>')[1] for i in ind_match]
        return rxns_list_with_reagents
    
    def get_conf_scores(self, probs_T3, ind_match): 
        '''
        Function returning the confidence scores of the forward validated reactions and the reactions themselves.

        --Inputs--
        rxns_list (list(str)):          List of forward validated reactions
        probs_T3 (list(float)):         List of confidence scores for each of the predicted products of 'preds_T3', in [0,1]
        ind_match (list(int)):          List of indices of the forward validated reactions

        --Returns--
        rxns_val (list(str)):           List of forward validated reactions (no modification from 'rxns_list')
        conf_scores (list(float)):      List of confidence scores for each of the forward validated reactions, in [0,1]
        '''
        conf_scores = [probs_T3[i] for i in ind_match]
        return conf_scores        

    def create_results_df(self):
        df = pd.DataFrame({'rxns': self.val_rxns_w_reagents_list, 'mapped_rxns': self.val_Map_rxns_w_reagents_list, 'conf_scores': self.conf_scores})
        return df
    
    def main(self):
        # exit if empty rxns_list
        if len(self.rxns_list) == 0:
            print("Empty reactions list")
            return

        # tokenize rxns_list
        self.tok_rxns_list = self.tokenize_rxn_list(self.rxns_list)

        # run T2 predictions
        self.T2_preds = self.run_T2_predictions(self.tok_rxns_list)

        # prepare reactions for T3
        self.rxns_format_T3_tok, self.mapped_reactions = self.format_rxns_for_T3(self.rxns_list, self.T2_preds)

        # run T3 predictions
        self.T3_preds, self.T3_probs = self.run_T3_predictions(self.rxns_format_T3_tok, self.T3_model_path)


        ind_match                          = self.get_match_indices(self.T3_preds, self.rxns_list)
        self.val_rxns_w_reagents_list      = self.add_reagents_to_rxns_list(self.rxns_list, self.T2_preds, ind_match)
        self.val_Map_rxns_w_reagents_list  = self.add_reagents_to_rxns_list(self.mapped_reactions, self.T2_preds, ind_match)
        self.conf_scores                   = self.get_conf_scores(self.T3_probs, ind_match)


        return self.create_results_df()