import numpy as np
import pytest
import tensorflow as tf
from src.model import ChurnModel

@pytest.fixture
def sample_data():
    """Create sample data for testing the model."""
    # Create random data
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 2, 20)
    X_test = np.random.rand(30, 10)
    y_test = np.random.randint(0, 2, 30)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def test_model_init():
    """Test model initialization."""
    model = ChurnModel(input_shape=10)
    
    # Check if model is a Sequential model
    assert isinstance(model.model, tf.keras.Sequential)
    
    # Check if model has the expected number of layers
    assert len(model.model.layers) == 5
    
    # Check if the input shape is correct
    assert model.model.layers[0].input_shape == (None, 10)
    
    # Check if the output layer has sigmoid activation
    assert model.model.layers[-1].activation.__name__ == 'sigmoid'

def test_model_compile():
    """Test if the model is compiled correctly."""
    model = ChurnModel(input_shape=10)
    
    # Check if the model is compiled
    assert model.model.optimizer is not None
    assert model.model.loss == 'binary_crossentropy'
    assert 'accuracy' in model.model.metrics_names

def test_model_train(sample_data):
    """Test model training."""
    X_train, y_train, X_val, y_val, _, _ = sample_data
    model = ChurnModel(input_shape=X_train.shape[1])
    
    # Train for just 2 epochs to speed up testing
    history = model.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=32)
    
    # Check if history object is returned
    assert isinstance(history, tf.keras.callbacks.History)
    
    # Check if history contains expected metrics
    assert 'loss' in history.history
    assert 'accuracy' in history.history
    assert 'val_loss' in history.history
    assert 'val_accuracy' in history.history
    
    # Check if we have the expected number of epochs in history
    assert len(history.history['loss']) == 2

def test_model_evaluate(sample_data):
    """Test model evaluation."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data
    model = ChurnModel(input_shape=X_train.shape[1])
    
    # Train for just 1 epoch to speed up testing
    model.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=32)
    
    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)
    
    # Check if evaluation returns loss and accuracy
    assert len(evaluation) == 2
    loss, accuracy = evaluation
    
    # Check if loss and accuracy are valid values
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
