import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.preprocessor import Preprocessor

@pytest.fixture
def sample_df():
    """
    Create a sample DataFrame for testing.
    """
    
    return pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Geography': ['France', 'Spain', 'Germany'],
        'Gender': ['Male', 'Female', 'Male'],
        'Age': [30, 35, 40],
        'Tenure': [5, 3, 8],
        'Balance': [50000, 0, 125000],
        'NumOfProducts': [2, 1, 3],
        'HasCrCard': [1, 0, 1],
        'IsActiveMember': [1, 1, 0],
        'EstimatedSalary': [100000, 80000, 120000],
        'Exited': [0, 0, 1]
    })

def test_preprocessor_init():
    """
    Test Preprocessor initialization.
    """
    
    preprocessor = Preprocessor()
    assert isinstance(preprocessor.encoders, dict)
    assert len(preprocessor.encoders) == 0
    assert isinstance(preprocessor.scaler, StandardScaler)

def test_rename_columns(sample_df):
    """
    Test column renaming functionality.
    """

    preprocessor = Preprocessor()
    renamed_df = preprocessor.rename_columns(sample_df)
    
    # Check if columns are renamed correctly
    expected_columns = ['customerid', 'geography', 'gender', 'age', 'active_years', 
                        'account_balance', 'num_of_products', 'credit_card', 
                        'active', 'salary_estimation', 'exited']
    
    assert list(renamed_df.columns) == expected_columns

def test_encode_features(sample_df):

    """Test categorical feature encoding."""
    preprocessor = Preprocessor()
    columns_to_encode = ['Geography', 'Gender']
    encoded_df = preprocessor.encode_features(sample_df, columns_to_encode)
    
    # Check if encoders are created for specified columns
    assert set(preprocessor.encoders.keys()) == set(columns_to_encode)
    
    # Check if encoded columns contain numeric values
    for col in columns_to_encode:
        assert encoded_df[col].dtype in [np.int32, np.int64]
    
    # Check if encoding is consistent
    geography_values = sample_df['Geography'].unique()
    for value in geography_values:
        # Find all rows with this value in the original DataFrame
        original_indices = sample_df[sample_df['Geography'] == value].index
        
        # Check if all these rows have the same encoded value in the encoded DataFrame
        encoded_value = encoded_df.loc[original_indices[0], 'Geography']
        for idx in original_indices:
            assert encoded_df.loc[idx, 'Geography'] == encoded_value

def test_scale_features():
    """
    Test feature scaling.
    """
    
    preprocessor = Preprocessor()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    scaled_X = preprocessor.scale_features(X)
    
    # Check if the scaled data has the same shape
    assert scaled_X.shape == X.shape
    
    # Check if the scaled data has mean close to 0 and std close to 1 for each feature
    assert np.allclose(scaled_X.mean(axis=0), np.zeros(X.shape[1]), atol=1e-10)
    assert np.allclose(scaled_X.std(axis=0), np.ones(X.shape[1]), atol=1e-10)
