import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    """
    Handles feature engineering and preprocessing
    """

    def __init__(self):

        self.encoders = {}
        self.scaler = StandardScaler()

    def rename_columns(self, df):

        return df.rename(columns = str.lower).rename(columns = {
            "tenure": "active_years", 
            "numofproducts": "num_of_products",
            "hascrcard": "credit_card",
            "rownumber": "index",
            "customerid": "customer_id",
            "creditscore": "credit_score",
            "isactivemember": "active",
            "estimatedsalary": "salary_estimation",
            "balance": "account_balance"
        })

    def encode_features(self, df, columns):

        for col in columns:

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        return df

    def scale_features(self, X):
        
        return self.scaler.fit_transform(X)