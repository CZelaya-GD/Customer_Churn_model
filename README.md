# Customer_Churn_model

## Overview

This project predicts customer churn using a neural network. It's designed to be modular, testable, and easy to integrate into a CI/CD pipeline for continuous improvement. The project follows an object-oriented approach to ensure code reusability and maintainability.

## Project Structure

customer-churn/
├── src/
│ ├── data_loader.py # Loads the dataset
│ ├── preprocessor.py # Handles data preprocessing
│ ├── model.py # Defines and trains the neural network
├── tests/
│ ├── test_data_loader.py   # Tests for data loading
│ ├── test_preprocessor.py   # Tests for preprocessing
│ └── test_model.py   # Tests for model training
├── notebooks/
│ └── Customer-churn-model.ipynb # Exploratory data analysis and model experimentation
├── requirements.txt   # Lists project dependencies
├── README.md   # This file
└── .github/
└── workflows/
└── ci.yml   # CI/CD pipeline configuration

## Modules

### `src/data_loader.py`

- **Purpose:** Loads the customer churn dataset from a CSV file.
- **Class:** `DataLoader`
  - `__init__(self, path)`: Initializes the DataLoader with the dataset path.
  - `load(self)`: Reads the CSV file into a pandas DataFrame and returns it.

### `src/preprocessor.py`

- **Purpose:** Preprocesses the data by renaming columns, encoding categorical features, and scaling numerical features.
- **Class:** `Preprocessor`
  - `__init__(self)`: Initializes the preprocessor with encoders and scalers.
  - `rename_columns(self, df)`: Renames columns to a standardized format (lowercase, snake_case).
  - `encode_features(self, df, columns)`: Encodes specified categorical columns using LabelEncoder.
  - `scale_features(self, X)`: Scales numerical features using StandardScaler.

### `src/model.py`

- **Purpose:** Defines, compiles, trains, and evaluates the neural network model.
- **Class:** `ChurnModel`
  - `__init__(self, input_shape)`: Initializes the ChurnModel with the input shape, builds the neural network.
  - `train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32)`: Trains the model.
  - `evaluate(self, X_test, y_test)`: Evaluates the trained model on the test dataset.

## Notebooks

### `notebooks/Customer-churn-model.ipynb`

- **Purpose:**
  - Performs exploratory data analysis (EDA) to understand the dataset.
  - Loads and preprocesses data using the modules in `src/`.
  - Trains and evaluates a neural network model for customer churn prediction.
  - Visualizes results and insights.

## Tests

### `tests/test_data_loader.py`

- Tests the data loading functionality of the `DataLoader` class.

### `tests/test_preprocessor.py`

- Tests the data preprocessing steps performed by the `Preprocessor` class.

### `tests/test_model.py`

- Tests the structure and basic functionality of the `ChurnModel` class.

## CI/CD Pipeline

### `.github/workflows/ci.yml`

- Configures a CI/CD pipeline using GitHub Actions to automate testing and notebook execution on every push and pull request.
- **Steps:**
  1. Checks out the code.
  2. Sets up Python.
  3. Installs dependencies from `requirements.txt`.
  4. Runs unit tests using `pytest`.
  5. Executes the notebook and checks for errors.

## Requirements

- Python 3.7+
- TensorFlow
- scikit-learn
- pandas
- numpy
- pytest
- nbconvert
- Jupyter

To install dependencies:

## Setup

1. Clone the repository.
2. Install the dependencies.
3. (Optional) Set up a virtual environment.

## Usage

1. Explore the dataset and experiment with the model in `notebooks/Customer-churn-model.ipynb`.
2. Run unit tests using `pytest tests/`.
3. Ensure the CI/CD pipeline is set up in your GitHub repository.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes.
4. Write tests for your changes.
5. Run all tests to ensure they pass.
6. Create a pull request.

## License

[Specify the license, e.g., MIT License]
