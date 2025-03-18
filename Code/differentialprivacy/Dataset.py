import pandas as pd
import numpy as np
from epsilon_DP import DifferentialPrivacy

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.dp = None
        self.epsilon = None
        self.sensitivity = 1
    def generate_random_data(self, num_rows=10000):
        self.data = pd.DataFrame({
            'ID': range(1, num_rows + 1),  # ID
            'salary': np.random.randint(2000, 12001, size=num_rows)  # Salary in range [2000, 12000]
        })
    
    def save_to_csv(self):
        if self.data is not None:
            self.data.to_csv(self.file_path, index=False)
            return True
        else:
            return False
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def set_epsilon(self, epsilon):
        if epsilon is None:
            self.dp = None
            self.epsilon = None
        else:
            self.epsilon = epsilon
            self.dp = DifferentialPrivacy(epsilon)
    
    def get_sensitive_column(self, column_name):
        if self.data is not None:
            return self.data[column_name].values
        else:
            return None
    
    def query(self):
        if self.dp is None:
            raw_data = self.get_sensitive_column('salary')
            return np.mean(raw_data)
        else:
            raw_data = self.get_sensitive_column('salary')
            return self.dp.laplace_mechanism(np.mean(raw_data), self.sensitivity)
            
        
    def add_data(self, individual_data):
        # Append the individual data to the dataset
        new_data = self.data._append(individual_data, ignore_index=True)
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        # Save the updated dataset to csv
        self.save_to_csv()
        return True
    
    def change_tail_data(self, column_name, new_value):
        self.data[column_name].iloc[-1] = new_value
        self.save_to_csv()
        return True

