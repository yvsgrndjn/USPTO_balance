import pandas as pd

class Dataset:
    def __init__(self, data=None, file_path=None, file_type='csv'):
        """
        Initialize the Dataset class.
        
        Parameters:
        data (pd.DataFrame or None): Preloaded data as DataFrame.
        file_path (str): Path to the dataset file.
        file_type (str): Type of the dataset file ('csv', 'json', 'excel', etc.)
        """
        self.data = None
        if data is not None:
            self.data = data
        elif file_path:
            self.load_data(file_path, file_type)
    
    def read_txt_to_list(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]

    def save_list_to_txt(self, file_path):
        with open(file_path, 'w') as f:
            for item in self.data:
                f.write(f"{item}\n")

    def load_data(self, file_path, file_type='csv'):
        """Load data from a file."""
        if file_type == 'csv':
            self.data = pd.read_csv(file_path)
        elif file_type == 'json':
            self.data = pd.read_json(file_path)
        elif file_type == 'excel':
            self.data = pd.read_excel(file_path)
        elif file_type == 'pkl':
            self.data = pd.read_pickle(file_path)
        elif file_type == 'txt':
            self.data = self.read_txt_to_list(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        print(f"Data loaded successfully from {file_path}")
    
    def basic_info(self):
        """Print basic information about the dataset."""
        if self.data is not None:
            print(self.data.info())
            print("\nHead of the Data:\n", self.data.head())
        else:
            print("No data loaded.")

    def clean_data(self, drop_duplicates=True, fill_missing=None):
        """Clean the dataset by handling missing values and duplicates."""
        if self.data is None:
            print("No data to clean.")
            return
        
        if drop_duplicates:
            self.data.drop_duplicates(inplace=True)
            print("Duplicates removed.")
        
        if fill_missing is not None:
            self.data.fillna(fill_missing, inplace=True)
            print(f"Missing values filled with {fill_missing}.")
    
    def filter_rows(self, condition):
        """
        Filter the dataset based on a condition.
        
        Parameters:
        condition (str): A condition string, e.g., 'column_name > 10'.
        
        Returns:
        pd.DataFrame: Filtered DataFrame.
        """
        if self.data is None:
            print("No data to filter.")
            return None
        return self.data.query(condition)
    
    def sample_data(self, n=5):
        """Return a random sample of the dataset."""
        if self.data is None:
            print("No data to sample.")
            return None
        return self.data.sample(n)

    def get_column(self, column_name):
        """Get a specific column from the dataset."""
        if self.data is None:
            print("No data loaded.")
            return None
        if column_name in self.data.columns:
            return self.data[column_name]
        else:
            print(f"Column '{column_name}' not found.")
            return None

    def save_data(self, output_path, file_type='csv'):
        """Save the dataset to a file."""
        if self.data is None:
            print("No data to save.")
            return
        if file_type == 'csv':
            self.data.to_csv(output_path, index=False)
        elif file_type == 'json':
            self.data.to_json(output_path)
        elif file_type == 'excel':
            self.data.to_excel(output_path, index=False)
        elif file_type == 'pkl':
            self.data.to_pickle(output_path)
        elif file_type == 'txt':
            self.data.save_list_to_txt(output_path)
            #  def save_list_to_txt(self, data, file_path):
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        print(f"Data saved to {output_path}")
    
    def __len__(self):
        """Return the number of rows in the dataset."""
        if self.data is not None:
            return len(self.data)
        return 0
    
    def __getitem__(self, key):
        """
        Get a row or column from the dataset.
        
        Parameters:
        key (int or str): If int, return a row by index. If str, return a column by name.
        
        Returns:
        pd.Series or pd.DataFrame: A row (as Series) or a column (as DataFrame).
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if isinstance(key, int):
            # If key is an integer, return the row at that index
            if key < 0 or key >= len(self.data):
                raise IndexError("Row index out of range.")
            return self.data.iloc[key]
        
        elif isinstance(key, str):
            # If key is a string, return the column with that name
            if key not in self.data.columns:
                raise KeyError(f"Column '{key}' not found.")
            return self.data[key]
        
        else:
            raise TypeError("Invalid key type. Use int for rows or str for columns.")

