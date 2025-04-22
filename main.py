import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import ChurnModel

def main():
    """
    Main function to execute the customer churn prediction workflow.
    """
    
    # 1. Load Data
    data_loader = DataLoader(r"Churn_Modelling.csv")  # Path to the dataset
    df = data_loader.load()

    # 2. Data Preprocessing
    preprocessor = Preprocessor()
    df = preprocessor.rename_columns(df)
    df = preprocessor.encode_features(df, columns=["geography", "gender"])

    # Prepare data for modeling
    X = df[['credit_score', 'age', 'active_years', 'account_balance', 
            'num_of_products', 'credit_card', 'active', 'salary_estimation']].values
    y = df['exited'].values
    X_scaled = preprocessor.scale_features(X)

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # 3. Model Training
    input_shape = X_train.shape[1]
    churn_model = ChurnModel(input_shape=input_shape)
    
    print("Training the model...")
    history = churn_model.train(X_train, y_train, X_val, y_val, epochs=50)

    # 4. Model Evaluation
    print("\nEvaluating the model...")
    loss, accuracy = churn_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nChurn prediction workflow completed.")

if __name__ == "__main__":
    main()
