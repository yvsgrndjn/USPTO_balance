from __future__ import division, unicode_literals
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import onmt.bin.preprocess as preprocess
import onmt.bin.train as train
import onmt.bin.translate as trsl

# for round trip accuracy function
import datetime
import pickle
import csv
from ttlretro.single_step_retro import SingleStepRetrosynthesis
singlestepretrosynthesis = SingleStepRetrosynthesis()
from ttlretro.rxnmarkcenter import RXNMarkCenter
rxnmarkcenter = RXNMarkCenter()
from rxnmapper import BatchedMapper
rxn_mapper_batch = BatchedMapper(batch_size=10, canonicalize=False)


# Folder structure
# the reference point to run calculations and find results is an OpenNMT-py folder, that can be accessed by the {path_to_folder} argument
# The folder structure will look like this 
#
#   OpenNMT-py
#       |--data
#           |--{dataset} (dataset used to preprocess and train the model, ex. USPTO_rand_1M)
#               |--{experiment} (ex. Tx_mmdd)
#                   |-- (files) src_train.txt , tgt_train.txt, ..
#               |--voc_{experiment} (where the vocabulary for a given model training will be stored)
#                   |-- (files) Preprocessed.train.0.pt , Preprocessed.valid.0.pt , Preprocessed.vocab.pt
#       |--experiments
#           |--checkpoints (contains the hyperparameters of the models at different steps)
#               |--{dataset} (dataset used to preprocess and train the model, ex. USPTO_rand_1M)
#                   |--{experiment} (ex. Tx_mmdd)
#                       |-- (files) {experiment}_model_step_x.pt
#           |--Tensorboard
#               |--{dataset} (dataset used to preprocess and train the model, ex. USPTO_rand_1M)
#                   |--{experiment}
#                       |--{Day-dd_hh_mm_ss}
#                           |-- (files) events.out.tfevents.xxxxxxxxxx.computername.xxxxxx.0
#       |--outputs
#           |--{dataset} 
#               |--{experiment}
#                   |-- (files) output_{experiment}_{step}.txt
#


# Data Preprocessing ----------

def preprocess_onmt_model(dataset: str, experiment: str, path_to_folder: str):
    '''
    Prepare vocabulary for model training. For that, needs to see the validation and training splits of the data to understand how the data is constructed and of which tokens.
    
    --Inputs:
    dataset(str):           name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
    experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
    path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
    
    --Outputs:
    saves files under ./data/{dataset}/voc_{experiment}/ containing vocabulary needed for the training of the model
    '''

    src_train_path = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/src_train.txt'
    tgt_train_path = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/tgt_train.txt'
    src_valid_path = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/src_val.txt'
    tgt_valid_path = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/tgt_val.txt'
    path_to_save_voc = f'{path_to_folder}OpenNMT-py/data/{dataset}/voc_{experiment}/Preprocessed'

    if not os.path.exists(f'{path_to_folder}OpenNMT-py/data/{dataset}/voc_{experiment}/'):
        os.makedirs(f'{path_to_folder}OpenNMT-py/data/{dataset}/voc_{experiment}/')

    src_seq_length = 3000
    tgt_seq_length = 3000
    src_vocab_size = 3000
    tgt_vocab_size = 3000

    args = [
        "-train_src", str(src_train_path),
        "-train_tgt", str(tgt_train_path),
        "-valid_src", str(src_valid_path),
        "-valid_tgt", str(tgt_valid_path),
        "-save_data", str(path_to_save_voc),
        "-src_seq_length", f"{src_seq_length}",
        "-tgt_seq_length", f"{tgt_seq_length}", 
        "-src_vocab_size", f"{src_vocab_size}", 
        "-tgt_vocab_size", f"{tgt_vocab_size}", 
        "-share_vocab",
        "-lower"
    ]

    parser = preprocess._get_parser()
    opt = parser.parse_args(args)
    preprocess.preprocess(opt)

    print(f'Preprocessing completed for experiment {experiment}')


# Train Model --------

def train_onmt_model(dataset:str, experiment:str, path_to_folder: str, train_steps:int =200000, batchsize:int =6144, dropout:float =0.1, rnnsize:int =384, wordvecsize:int =384, learnrate:int =2):
    '''
    Train transformer model with the onmt package.

    --Inputs
    dataset(str):           name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
    experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
    path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
    
    --Outputs
    Saves last "-keep_checkpoint" (20 by default) models in .pt files under /OpenNMT-py/experiments/checkpoints/{dataset}/{experiment}/{experiment}_model
    Tensorboard files are saved under OpenNMT-py/experiments/Tensorboard/{dataset}/{experiment}/
    '''

    args = [
    "-data",                        f"{path_to_folder}OpenNMT-py/data/{dataset}/voc_{experiment}/Preprocessed",
    "-save_model",                  f"{path_to_folder}OpenNMT-py/experiments/checkpoints/{dataset}/{experiment}/{experiment}_model",
    "-seed",                        "42",
    "-save_checkpoint_steps",       "5000",
    "-keep_checkpoint",             "20",
    "-train_steps",                 f"{train_steps}",
    "-param_init",                  "0",
    "-param_init_glorot",
    "-max_generator_batches",       "32",
    "-batch_size",                  f"{batchsize}",
    "-batch_type",                  "tokens",
    "-normalization",               "tokens",
    "-max_grad_norm",               "0",
    "-accum_count",                 "4",
    "-optim",                       "adam",
    "-adam_beta1",                  "0.9",
    "-adam_beta2",                  "0.998",
    "-decay_method",                "noam",
    "-warmup_steps",                "8000",
    "-learning_rate",               f"{learnrate}",
    "-label_smoothing",             "0.0",
    "-layers",                      "4",
    "-rnn_size",                    f"{rnnsize}",
    "-word_vec_size",               f"{wordvecsize}",
    "-encoder_type",                "transformer",
    "-decoder_type",                "transformer",
    "-dropout",                     f"{dropout}",
    "-position_encoding",
    "-global_attention",            "general",
    "-global_attention_function",   "softmax",
    "-self_attn_type",              "scaled-dot",
    "-heads",                       "8",
    "-transformer_ff",              "2048",
    "-valid_steps",                 "5000",
    "-valid_batch_size",            "4",
    "-report_every",                "1000",
    "-log_file",                    f"{path_to_folder}OpenNMT-py/data/{dataset}/Training_LOG_{experiment}.txt",
    "-early_stopping",              "10",
    "-early_stopping_criteria",     "accuracy",
    "-world_size",                  "1",
    "-gpu_ranks",                   "0",
    "-tensorboard",
    "-tensorboard_log_dir",         f"{path_to_folder}OpenNMT-py/experiments/Tensorboard/{dataset}/{experiment}/"
    ]

    parser = train._get_parser()
    opt = parser.parse_args(args)
    train.train(opt)
    
    print(f'Training done for experiment {experiment}')


# Translate (=inference of the model)-----------

def translate_with_onmt_model(dataset:str, experiment:str, path_to_folder: str, step:int, src_path: str, data_inference: str, beam_size:int =3, batch_size:int  = 64):
    '''
    Model inference.

    --Inputs
    dataset (str):          name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
    experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
    path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
    step (int):             model step of the model to use for inference 
    src_path (str):         path of the src_test.txt used for inference
    data_inference (str):   name of the dataset at the origin of the src_test.txt for inference
    beam_size (int):        (default 3) number of different predictions the model will make for a given query
    batch_size (int):       (default 64) number of queries performed at the same time for inference

    --Outputs
    List of predictions in txt file (beam_size) x longer as src_test.txt stored under f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/output_{experiment}_{step}.txt'
    '''

    Model_path  = f'{path_to_folder}OpenNMT-py/experiments/checkpoints/{dataset}/{experiment}/{experiment}_model_step_{step}.pt'
    #input_file  = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/src_test.txt'
    input_file  = src_path
    output_file = f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/output_{experiment}_{step}_on_{data_inference}.txt'

    if not os.path.exists(f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/'):
        os.makedirs(f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/')

    args = [
        "-beam_size",   str(beam_size), 
        "-n_best",      str(beam_size), 
        "-model",       str(Model_path), 
        "-src",         str(input_file), 
        "-output",      str(output_file), 
        "-batch_size",  str(batch_size), 
        "-max_length",  "1000", 
        "-log_probs",
        "-replace_unk"
    ]

    parser = trsl._get_parser()
    opt = parser.parse_args(args)
    trsl.translate(opt)


# EVALUATE -------

def canonicalize_smiles(smiles:str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['Target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


def get_rank_RTA(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['{}{}'.format(base, i)]:
            return i
    return 0


def evaluate_onmt_model(src_output_path:str, tgt_path:str, path_to_folder:str, dataset:str, experiment:str, step:int, beam_size:int =3, data_test_name: str = '')-> pd.DataFrame:
    '''
    Compares the inference on the test set (src_test.txt) and the ground truth on it (tgt_test.txt) for a given beam-size and model evaluation.
    Returns accuracies from top-1 to top-(beam_size).

    --Inputs
    src_output_path(str):           Path to the output of the inference of src_test.txt through model {experiment} file that we want to evaluate 
    tgt_path(str):                  Path to the ground truth tgt_test.txt file against which the output will be compared.
    path_to_folder(str):            Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
    dataset(str):                   Name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
    experiment(str):                Name of the specific experiment done on the splits, usually the name of the model we will be training/using (such as Tx_mmdd)
    step(int):                      model step of the model to use/used for inference
    beam_size(int):                 (default 3) number of different predictions the model will make for a given query
    data_test_name(str):            Name of the data on which inference was run (only used to save the dataframe afterwards)

    --Outputs
    test_df (pd.DataFrame)          dataframe containing the output for the different beam sizes and the according ground truth as given by the tgt_test.txt file
                                    is saved under f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/eval_df_output_{experiment}_{step}_{data_test_name}.csv'
    '''
    test_df = ""

    #src_file = f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/output_{experiment}_{step}.txt'
    src_file = src_output_path
    #tgt_file = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/tgt_test.txt' # should it be linked with the src_file? or directly tgt_path?
    tgt_file = tgt_path

    predictions = [[] for i in range(beam_size)]
    with open(src_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % beam_size].append(''.join(line.strip().split(' ')))
    probs = [[] for i in range(beam_size)]
    with open(src_file + '_log_probs', 'r') as f:
        for i, line in enumerate(f.readlines()):
            probs[i % beam_size].append(''.join(line.strip().split(' ')))
    target = []
    with open(tgt_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            target.append(''.join(line.strip().split(' ')))

    print('start canonicalizing...')
    
    
    test_df = pd.DataFrame(['' for element in range(0, len(predictions[0]))])#targets)
    test_df.columns = ['Target']

    for i, preds in tqdm(enumerate(predictions)):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(lambda x: canonicalize_smiles(x))
    for i, tgt in tqdm(enumerate(target)):
        test_df['Target'][i] = canonicalize_smiles(tgt)
    for i, prob in tqdm(enumerate(probs)):
        test_df['prediction_prob_{}'.format(i + 1)] = prob

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', beam_size), axis=1)

    if beam_size == 3:
        for element in range(0, len(test_df)):
            test_df['prediction_prob_1'][element] = 10**(float(test_df['prediction_prob_1'][element]))
            test_df['prediction_prob_2'][element] = 10**(float(test_df['prediction_prob_2'][element]))
            test_df['prediction_prob_3'][element] = 10**(float(test_df['prediction_prob_3'][element]))
    elif beam_size == 5:
        for element in range(0, len(test_df)):
            test_df['prediction_prob_1'][element] = 10**(float(test_df['prediction_prob_1'][element]))
            test_df['prediction_prob_2'][element] = 10**(float(test_df['prediction_prob_2'][element]))
            test_df['prediction_prob_3'][element] = 10**(float(test_df['prediction_prob_3'][element]))
            test_df['prediction_prob_4'][element] = 10**(float(test_df['prediction_prob_4'][element]))
            test_df['prediction_prob_5'][element] = 10**(float(test_df['prediction_prob_5'][element]))
        
    print(' ')  
    #print('PROJECT REF:', src_file.split('prediction_')[1].split('_')[0])
    print(' ')  
    test_df.to_csv(f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/eval_df_output_{experiment}_{step}_on_{data_test_name}.csv')

    correct = 0
    total = len(test_df)
    for i in range(1, beam_size+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles = (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        if True: print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,invalid_smiles/total*100))
        else: print('Top-{}: {:.1f}%'.format(i, correct / total * 100))

    return test_df


def evaluate_onmt_model_template_dependent(src_output_path:str, tgt_path:str, path_to_folder:str, dataset:str, experiment:str, step:int, beam_size:int =3, data_test_name: str = '', list_template_lines:list = [])-> pd.DataFrame:
    '''
    Compares the inference on the test set (src_test.txt) and the ground truth on it (tgt_test.txt) for a given beam-size and model evaluation.
    Calculates the top-k accuracies for each template and adds them to the output dataframe.
    Returns (unweighted) average template accuracy from top-1 to top-(beam_size).

    --Inputs
    src_output_path(str):           Path to the output of the inference of src_test.txt through model {experiment} file that we want to evaluate 
    tgt_path(str):                  Path to the ground truth tgt_test.txt file against which the output will be compared.
    path_to_folder(str):            Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
    dataset(str):                   Name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
    experiment(str):                Name of the specific experiment done on the splits, usually the name of the model we will be training/using (such as Tx_mmdd)
    step(int):                      model step of the model to use/used for inference
    beam_size(int):                 (default 3) number of different predictions the model will make for a given query
    data_test_name(str):            Name of the data on which inference was run (only used to save the dataframe afterwards)

    --Outputs
    test_df (pd.DataFrame)          dataframe containing the output for the different beam sizes and the according ground truth as given by the tgt_test.txt file
                                    is saved under f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/eval_df_output_{experiment}_{step}_{data_test_name}.csv'
    '''
    test_df = ""

    #src_file = f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/output_{experiment}_{step}.txt'
    src_file = src_output_path
    #tgt_file = f'{path_to_folder}OpenNMT-py/data/{dataset}/{experiment}/tgt_test.txt' # should it be linked with the src_file? or directly tgt_path?
    tgt_file = tgt_path

    predictions = [[] for i in range(beam_size)]
    with open(src_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % beam_size].append(''.join(line.strip().split(' ')))
    probs = [[] for i in range(beam_size)]
    with open(src_file + '_log_probs', 'r') as f:
        for i, line in enumerate(f.readlines()):
            probs[i % beam_size].append(''.join(line.strip().split(' ')))
    target = []
    with open(tgt_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            target.append(''.join(line.strip().split(' ')))

    # stop calculations right away if the predictions and target lengths do not match
    if len(target) != len(predictions[0]):
        print('len target: ', len(target))
        print('len(predictions): ', len(predictions[0]))
        print('LengthMismatchError: length of predictions and target do not match')

    print('start canonicalizing...')
    
    
    test_df = pd.DataFrame(['' for element in range(0, len(predictions[0]))])#targets)
    test_df.columns = ['Target']

    for i, preds in tqdm(enumerate(predictions)):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(lambda x: canonicalize_smiles(x))
    for i, tgt in tqdm(enumerate(target)):
        test_df['Target'][i] = canonicalize_smiles(tgt)
    for i, prob in tqdm(enumerate(probs)):
        test_df['prediction_prob_{}'.format(i + 1)] = prob

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', beam_size), axis=1)

    if beam_size == 3:
        for element in range(0, len(test_df)):
            test_df['prediction_prob_1'][element] = 10**(float(test_df['prediction_prob_1'][element]))
            test_df['prediction_prob_2'][element] = 10**(float(test_df['prediction_prob_2'][element]))
            test_df['prediction_prob_3'][element] = 10**(float(test_df['prediction_prob_3'][element]))
    elif beam_size == 5:
        for element in range(0, len(test_df)):
            test_df['prediction_prob_1'][element] = 10**(float(test_df['prediction_prob_1'][element]))
            test_df['prediction_prob_2'][element] = 10**(float(test_df['prediction_prob_2'][element]))
            test_df['prediction_prob_3'][element] = 10**(float(test_df['prediction_prob_3'][element]))
            test_df['prediction_prob_4'][element] = 10**(float(test_df['prediction_prob_4'][element]))
            test_df['prediction_prob_5'][element] = 10**(float(test_df['prediction_prob_5'][element]))
        
    print(' ')  
    #print('PROJECT REF:', src_file.split('prediction_')[1].split('_')[0])
    print(' ')  

    for el in range(1, beam_size+1):
        test_df['top{}_temp'.format(el)] = 0

    if len(list_template_lines) == len(test_df):
        test_df['template_line'] = list_template_lines
    else: 
        print('LengthError: length of template lines numbers does not match the length of the output')
        
    # Group by 'template_line' and calculate correct and invalid_smiles
    grouped = test_df.groupby('template_line')
    for name, group in grouped:
        total = len(group)
        if total == 0:
            continue
        correct = np.zeros(beam_size)
        invalid_smiles = np.zeros(beam_size)
        for i in range(1, beam_size+1):
            correct[i-1] = (group['rank'] == i).sum()
            invalid_smiles[i-1] = (group['canonical_prediction_{}'.format(i)] == '').sum()
        accuracy = np.cumsum(correct / total * 100)
        for i in range(beam_size):
            test_df.loc[group.index, 'top{}_temp'.format(i+1)] = accuracy[i]
    
    test_df.to_csv(f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/eval_df_tempdep_output_{experiment}_{step}_on_{data_test_name}.csv')
    
    # Calculate the (unweighted) average template accuracy
    for i in range(1, beam_size+1):
        average_acc = sum(grouped['top{}_temp'.format(i)].mean())/len(grouped)
        print('Top-{}: {:.3f}%'.format(i, average_acc))

    return test_df


def RoundTrip(
    SRC_TEST_SET_PATH:str, 
    MODEL_T1_PATH:str, 
    MODEL_T2_PATH:str, 
    MODEL_T3_PATH:str, 
    MODEL_T3_FT_PATH:str, 
    T2_INPUT_LABEL:str, 
    USE_T3_FT:bool=False, 
    save_pickle_path:str='./save.pkl', 
    Max_SRC_Length:int=100000, 
    random_state:int=42
    ):
    '''
    Calculates the round trip accuracy (RTA) of the TTL and TTLFT model types. For a reaction described as A>B>C, takes
    as input tok(C!). The retrosynthesis model (T1) will predict tok(A'). Reagent prediction model (T2) takes tok(A') > > tok(C)
    as input and returns tok(B'). The boolean 'USE_T3_FT' parameter allows to set if the forward validation model (T3)
    works with tagged reactants (=Forward Tag) or not (denoted T3FT). Forward validation model (T3) takes tok(A') > tok(B')
    as input and returns C'.
    The round trip accuracy is calculated as the sum^i_0( C_i == C'_i )/i
    The reagent prediction model (T2) is set to produce three suggestions for each Retrosynthetic prediction (=beam size of 3), 
    giving the top-k accuracies with k = 1,2,3.

    --Input
    SRC_TEST_SET_PATH (str) :
    MODEL_T1_PATH     (str) :    
    MODEL_T2_PATH     (str) :    
    MODEL_T3_PATH     (str) :    
    MODEL_T3_FT_PATH  (str) : 
    T2_INPUT_LABEL    (str) :   
    USE_T3_FT         (bool):      
    save_pickle_path  (str) : 
    Max_SRC_Length    (int) :   
    random_state      (int) :     

    --Output
    '''

    CurrentParameters = {
        'SRC_TEST_SET_PATH' : SRC_TEST_SET_PATH, 
        'MODEL_T1_PATH'     : MODEL_T1_PATH, 
        'MODEL_T2_PATH'     : MODEL_T2_PATH, 
        'MODEL_T3_PATH'     : MODEL_T3_PATH, 
        'MODEL_T3_FT_PATH'  : MODEL_T3_FT_PATH,
        'T2_INPUT_LABEL'    : T2_INPUT_LABEL,
        'USE_T3_FT'         : USE_T3_FT,
        'save_pickle_path'  : save_pickle_path, 
        'Max_SRC_Length'    : Max_SRC_Length,
        'random_state'      : random_state
    }
    src_test = []
    src_test_tagged = []
    with open(SRC_TEST_SET_PATH, 'r') as f:
        for i, line in enumerate(f.readlines()):
            src_test.append(''.join(line.strip().split(' ')).replace('!',''))
            src_test_tagged.append(line.strip())

    df = pd.DataFrame(['' for el in range(0, len(src_test))])
    df.attrs['ModelParameters'] = CurrentParameters
    df.columns                  = ['SRC_T1']
    df['SRC_smiles']            = src_test
    df['SRC_T1_input']          = src_test_tagged

    # Sample dataframe if length exceeds the maximum allowed:
    if len(df) > Max_SRC_Length:
        df = df.sample(n=Max_SRC_Length, random_state=random_state).reset_index(drop=True)
    print('Length:', len(df))
    df.to_pickle(save_pickle_path)

    # Check for tmp folder existence needed for onmt predictions
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    #Part 2 -- T1 predictions
    #-------
    print('Predicting T1...')
    predictions, probs = singlestepretrosynthesis.Execute_Prediction(
        SMILES_list         = df['SRC_T1_input'].tolist(),
        Model_path          = MODEL_T1_PATH, 
        beam_size           = 5, 
        batch_size          = 64, 
        untokenize_output   = False, 
    )
    df['T1_pred_1']         = predictions[0]
    df['T1_pred_conf_1']    = probs[0]
    df.to_pickle(save_pickle_path)

    # PREDICTIONS THAT DID NOT DO ANYTHING:
    df['T1_pred_1_check']   = [''.join(el.split(' ')) for el in df['T1_pred_1']]
    df['T1_pred_1_check2']  = False
    for el in range(0, len(df)):
        if df.at[el, 'SRC_smiles'] in df.at[el, 'T1_pred_1_check'].split('.'):
            df.at[el, 'T1_pred_1_check2'] = True
    del df['T1_pred_1_check']

    #Part 3 -- T2 predictions
    #-------
    print('Predicting T2...')
    df['SRC_T2_input'] = [df.at[el, 'T1_pred_1'] + ' > > ' + df.at[el, 'SRC_T1_input'].replace(' !', '') + T2_INPUT_LABEL for el in range(0, len(df))]

    predictions, probs = singlestepretrosynthesis.Execute_Prediction(
        SMILES_list         = df['SRC_T2_input'].tolist(),
        Model_path          = MODEL_T2_PATH, 
        beam_size           = 3, 
        batch_size          = 64, 
        untokenize_output   = False, 
    )
    df['T2_pred_1'] = predictions[0]
    df['T2_pred_2'] = predictions[1]
    df['T2_pred_3'] = predictions[2]
    df.to_pickle(save_pickle_path)


    #Part 4 -- T3 predictions + sumup 
    #-------
    if not USE_T3_FT:
        print('Predicting T3, first reagent set...')
        df['SRC_T3_input_R1'] = [df.at[el, 'T1_pred_1'] + ' > ' + df.at[el, 'T2_pred_1'] for el in range(0, len(df))]
        df['SRC_T3_input_R2'] = [df.at[el, 'T1_pred_1'] + ' > ' + df.at[el, 'T2_pred_2'] for el in range(0, len(df))]
        df['SRC_T3_input_R3'] = [df.at[el, 'T1_pred_1'] + ' > ' + df.at[el, 'T2_pred_3'] for el in range(0, len(df))]

        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list         = df['SRC_T3_input_R1'].tolist(),
            Model_path          = MODEL_T3_PATH, 
            beam_size           = 3, 
            batch_size          = 64, 
            untokenize_output   = True, 
        )
        df['T3_pred_R1']        = predictions[0]
        df['T3_pred_conf_R1']   = probs[0]
        df.to_pickle(save_pickle_path)

        print('Predicting T3, second reagent set...')
        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list         = df['SRC_T3_input_R2'].tolist(),
            Model_path          = MODEL_T3_PATH, 
            beam_size           = 3, 
            batch_size          = 64, 
            untokenize_output   = True, 
        )
        df['T3_pred_R2']        = predictions[0]
        df['T3_pred_conf_R2']   = probs[0]
        df.to_pickle(save_pickle_path)

        print('Predicting T3, third reagent set...')
        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list         = df['SRC_T3_input_R3'].tolist(),
            Model_path          = MODEL_T3_PATH, 
            beam_size           = 3, 
            batch_size          = 64, 
            untokenize_output   = True, 
        )

        df['T3_pred_R3']        = predictions[0]
        df['T3_pred_conf_R3']   = probs[0]
        df.to_pickle(save_pickle_path)


        df['SRC_smiles_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['SRC_smiles']]
        df['T3_pred_R1_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_pred_R1']]
        df['T3_pred_R2_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_pred_R2']]
        df['T3_pred_R3_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_pred_R3']]

        df['T3_pred_R1_acc'] = [True if df.at[el, 'T3_pred_R1_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]
        df['T3_pred_R2_acc'] = [True if df.at[el, 'T3_pred_R2_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]
        df['T3_pred_R3_acc'] = [True if df.at[el, 'T3_pred_R3_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]

        df['T3_pred_R1_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_pred_R1_acc'] for el in range(0, len(df))]
        df['T3_pred_R2_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_pred_R2_acc'] for el in range(0, len(df))]
        df['T3_pred_R3_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_pred_R3_acc'] for el in range(0, len(df))]

        print('\n')
        print(round(100*len(df[(df['T3_pred_R1_acc'] == True)]), 2) / len(df), '% top-1 T2 reagents')
        print(round(100*len(df[(df['T3_pred_R2_acc'] == True) | (df['T3_pred_R1_acc'] == True)]), 2) / len(df), '% top-2 T2 reagents')
        print(round(100*len(df[(df['T3_pred_R3_acc'] == True) | (df['T3_pred_R2_acc'] == True) | (df['T3_pred_R1_acc'] == True)]), 2) / len(df), '% top-3 T2 reagents')
        print('\n')

        df.to_pickle(save_pickle_path)
    else: # part 5 -- T3 Forward tag predictions + sumup
        print('Mapping reactions...')
        
        # prepare A>>C reactions untokenized for mapping
        df['rxnstomap']      = [df.at[el, 'T1_pred_1'].replace(' ','') + '>>' + df.at[el, 'SRC_smiles'] for el in range(0, len(df))]
        df['MappedReaction'] = list(rxn_mapper_batch.map_reactions(df['rxnstomap'].to_list()))
        df['TaggedReaction'] = ['' if df.at[el, 'MappedReaction']=='>>' else rxnmarkcenter.TagMappedReactionCenter(df.at[el, 'MappedReaction'], alternative_marking=True, tag_reactants=True) for el in range(len(df))]

        print('Predicting T3 with tagged reactants, first reagent set...')
        df['SRC_T3_FT_input_R1'] = ['' if df.at[el, 'TaggedReaction']=='' else singlestepretrosynthesis.smi_tokenizer(df.at[el, 'TaggedReaction'].split('>>')[0]) + ' > ' + df.at[el, 'T2_pred_1'] for el in range(0, len(df))]
        df['SRC_T3_FT_input_R2'] = ['' if df.at[el, 'TaggedReaction']=='' else singlestepretrosynthesis.smi_tokenizer(df.at[el, 'TaggedReaction'].split('>>')[0]) + ' > ' + df.at[el, 'T2_pred_2'] for el in range(0, len(df))]
        df['SRC_T3_FT_input_R3'] = ['' if df.at[el, 'TaggedReaction']=='' else singlestepretrosynthesis.smi_tokenizer(df.at[el, 'TaggedReaction'].split('>>')[0]) + ' > ' + df.at[el, 'T2_pred_3'] for el in range(0, len(df))]
        del df['rxnstomap']
        del df['MappedReaction']
        
        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list         = df['SRC_T3_FT_input_R1'].tolist(),
            Model_path          = MODEL_T3_FT_PATH, 
            beam_size           = 3, 
            batch_size          = 64, 
            untokenize_output   = True, 
        )
        df['T3_FT_pred_R1']      = predictions[0]
        df['T3_FT_pred_conf_R1'] = probs[0]
        df.to_pickle(save_pickle_path)

        print('Predicting T3 with tagged reactants, second reagent set...')
        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list        = df['SRC_T3_FT_input_R2'].tolist(),
            Model_path         = MODEL_T3_FT_PATH, 
            beam_size          = 3, 
            batch_size         = 64, 
            untokenize_output  = True, 
        )
        df['T3_FT_pred_R2']      = predictions[0]
        df['T3_FT_pred_conf_R2'] = probs[0]
        df.to_pickle(save_pickle_path)

        print('Predicting T3 with tagged reactants, third reagent set...')
        predictions, probs = singlestepretrosynthesis.Execute_Prediction(
            SMILES_list       = df['SRC_T3_FT_input_R3'].tolist(),
            Model_path        = MODEL_T3_FT_PATH, 
            beam_size         = 3, 
            batch_size        = 64, 
            untokenize_output = True, 
        )
        df['T3_FT_pred_R3']      = predictions[0]
        df['T3_FT_pred_conf_R3'] = probs[0]
        df.to_pickle(save_pickle_path)


        df['SRC_smiles_can']    = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['SRC_smiles']]
        df['T3_FT_pred_R1_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_FT_pred_R1']]
        df['T3_FT_pred_R2_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_FT_pred_R2']]
        df['T3_FT_pred_R3_can'] = [singlestepretrosynthesis.canonicalize_smiles(el) for el in df['T3_FT_pred_R3']]

        df['T3_FT_pred_R1_acc'] = [True if df.at[el, 'T3_FT_pred_R1_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]
        df['T3_FT_pred_R2_acc'] = [True if df.at[el, 'T3_FT_pred_R2_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]
        df['T3_FT_pred_R3_acc'] = [True if df.at[el, 'T3_FT_pred_R3_can'] == df.at[el, 'SRC_smiles_can'] else False for el in range(0, len(df))]

        df['T3_FT_pred_R1_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_FT_pred_R1_acc'] for el in range(0, len(df))]
        df['T3_FT_pred_R2_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_FT_pred_R2_acc'] for el in range(0, len(df))]
        df['T3_FT_pred_R3_acc'] = [False if df.at[el, 'T1_pred_1_check2'] == True else df.at[el, 'T3_FT_pred_R3_acc'] for el in range(0, len(df))]


        print('\n')
        print(round(100*len(df[(df['T3_FT_pred_R1_acc'] == True)]), 2) / len(df), '% top-1 T2 reagents')
        print(round(100*len(df[(df['T3_FT_pred_R2_acc'] == True) | (df['T3_FT_pred_R1_acc'] == True)]), 2) / len(df), '% top-2 T2 reagents')
        print(round(100*len(df[(df['T3_FT_pred_R3_acc'] == True) | (df['T3_FT_pred_R2_acc'] == True) | (df['T3_FT_pred_R1_acc'] == True)]), 2) / len(df), '% top-3 T2 reagents')
        print('\n')

        df.to_pickle(save_pickle_path)

    return df 


def RoundTrip_template_dependent(
    df_result_RTA: pd.DataFrame,
    list_template_lines: list,
    beam_size: int = 3
    ):
    '''
    ...
    '''
    if len(list_template_lines) == len(df_result_RTA):
        df_result_RTA['template_line'] = list_template_lines
    else: 
        print('LengthError: length of template lines numbers does not match the length of the output')

    # Group by 'template_line' and calculate correct and invalid_smiles
    grouped = df_result_RTA.groupby('template_line')

    for name, group in grouped:
        total = len(group)
        if total == 0:
            continue
        correct = np.zeros(beam_size)
        #invalid_smiles = np.zeros(beam_size)
        for i in range(1, beam_size+1):
            correct[i-1] = (group['rank'] == i).sum()
            #invalid_smiles[i-1] = (group['canonical_prediction_{}'.format(i)] == '').sum()
        accuracy = np.cumsum(correct / total * 100)
        for i in range(beam_size):
            df_result_RTA.loc[group.index, 'top{}_temp'.format(i+1)] = accuracy[i]
    
    #df_result_RTA.to_pickle(save_path)

    # Calculate the (unweighted) average template accuracy
    for i in range(1, beam_size+1):
        average_acc = sum(grouped['top{}_temp'.format(i)].mean())/len(grouped)
        print('Top-{}: {:.3f}%'.format(i, average_acc))