import pandas as pd

class DataLoader:
    """
    Loads and provides access to the churn dataset.
    """
    
    def __init__(self, path):
        
        self.path = path
        self.df = None

    def load(self):
        
        self.df = pd.read_csv(self.path)
        return self.df
