

class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        with open(self.data_path, 'r') as file:
            data = file.read()
        return data