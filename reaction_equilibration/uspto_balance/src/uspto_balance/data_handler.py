import pandas as pd
from src.utils.helper_functions import find_input_type

class DataHandler:
    def __init__(self, file_path):
        self.file_path  = file_path
        self.datatype   = find_input_type(file_path)

    def load_data(self): 
        if self.datatype == 'csv':
            return self._load_csv_data()
        
        elif self.datatype == 'json':
            return self._load_json_data()
        
        elif self.datatype == 'txt':
            return self._load_txt_data()
        
        elif self.datatype == 'pkl':
            return self._load_pkl_data()
        
        else:
            raise ValueError('File type not supported')
    
    def _load_csv_data(self):
        try:
            return pd.read_csv(self.file_path)
        
        except Exception as e:
            raise ValueError('Error loading csv file:', e)
    
    def _load_json_data(self):
        try:
            return pd.read_json(self.file_path)
        
        except Exception as e:
            raise ValueError('Error loading json file:', e)

    def _load_txt_data(self):
        try:
            with open(self.file_path, 'r') as file:
                return file.readlines()
        
        except Exception as e:
            raise ValueError('Error loading txt file:', e)
    
    def _load_pkl_data(self):
        try:
            return pd.read_pickle(self.file_path)
        
        except Exception as e:
            raise ValueError('Error loading pkl file:', e)