from __future__ import division, unicode_literals
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import sys, getopt
import pandas as pd
import pickle
import csv
import datetime
from uspto_balance import execute_onmt


def main(argv):
    
    SRC_TEST_SET_PATH   = ''
    MODEL_T1_PATH       = ''
    MODEL_T2_PATH       = ''
    MODEL_T3_PATH       = ''
    MODEL_T3_FT_PATH    = ''
    T2_INPUT_LABEL      = ''
    USE_T3_FT           = ''
    save_pickle_path    = ''
    Max_SRC_Length      = 150000
    random_state        = 42
    
    HELP = ' '.join([
        'Round_Trip_Accuracy.py', 
        '-src <src_test_set>', 
        '--modelT1 <modelT1_path>', 
        '--modelT2 <modelT2_path>', 
        '--modelT3 <modelT3_path>', 
        '--modelT3FT <modelT3FT_path>', 
        '--T2_Label <T2_input_label>', 
        '--useT3FT <use_T3_FT>', 
        '--pickle_path <save_pickle_path>', 
        '--Max_SRC_Length <Max_SRC_Length>'
        '--random_state <random_state>'
    ])
    
    try:
        opts, _ = getopt.getopt(argv,"hs:",["src=","modelT1=","modelT2=","modelT3=","modelT3FT=","T2_Label=","useT3FT=","pickle_path=", "Max_SRC_Length="])
    except getopt.GetoptError:
        print(HELP)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(HELP)
            sys.exit(2)
        elif opt in ("-s", "--src"):
            SRC_TEST_SET_PATH = arg
        elif opt in ("--modelT1"):
            MODEL_T1_PATH = arg
        elif opt in ("--modelT2"):
            MODEL_T2_PATH = arg
        elif opt in ("--modelT3"):
            MODEL_T3_PATH = arg
        elif opt in ("--modelT3FT"):
            MODEL_T3_FT_PATH = arg
        elif opt in ("--T2_Label"):
            T2_INPUT_LABEL = arg
        elif opt in ("--useT3FT"):
            USE_T3_FT = arg
        elif opt in ("--pickle_path"):
            save_pickle_path = arg + '_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '.pkl'
        elif opt in ("--Max_SRC_Length"):
            Max_SRC_Length = int(arg)
        elif opt in ("--random_state"):
            random_state = int(arg)
            
    # if any argument is missing, exit:
    if SRC_TEST_SET_PATH == '' or MODEL_T1_PATH == '' or MODEL_T2_PATH == '' or MODEL_T3_PATH == '':
        print("\n Missing argument(s):\n\n" + HELP + "\n")
        sys.exit(2)
    # take care of USE_T3_FT
    if USE_T3_FT=='False' or USE_T3_FT=='false':
        USE_T3_FT = False
    elif USE_T3_FT=='True' or USE_T3_FT=='true':
        USE_T3_FT = True
    else:
        print('\n Invalid value for --useT3FT argument, should be boolean:\n\n' + HELP + "\n" )
            
    execute_onmt.RoundTrip(
        SRC_TEST_SET_PATH   = SRC_TEST_SET_PATH, 
        MODEL_T1_PATH       = MODEL_T1_PATH, 
        MODEL_T2_PATH       = MODEL_T2_PATH, 
        MODEL_T3_PATH       = MODEL_T3_PATH,
        MODEL_T3_FT_PATH    = MODEL_T3_FT_PATH, 
        T2_INPUT_LABEL      = T2_INPUT_LABEL, 
        USE_T3_FT           = USE_T3_FT, 
        save_pickle_path    = save_pickle_path, 
        Max_SRC_Length      = Max_SRC_Length, 
        random_state        = random_state
        )


if __name__ == '__main__':
    main(sys.argv[1:])