import pytest
import pandas as pd
from src.data_loader import DataLoader

def test_data_loader_loads_data():
    """
    Test that DataLoader correctly loads data into a DataFrame.
    """
    
    test_csv_path = "test_data.csv"
    test_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    test_data.to_csv(test_csv_path, index=False)

    data_loader = DataLoader(test_csv_path)
    df = data_loader.load()

    assert isinstance(df, pd.DataFrame), "load() should return a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) == 2, "DataFrame should have the correct number of rows"
    assert list(df.columns) == ['col1', 'col2'], "DataFrame should have the correct columns"

    # Cleanup: Remove the test CSV file
    import os
    os.remove(test_csv_path)