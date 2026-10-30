"""
data_loader.py
Loads the Flipkart dataset and performs basic Exploratory Data Analysis (EDA).
"""

import pandas as pd


def load_data(file_path):
    """
    Load CSV dataset from the given file path.
    Returns the DataFrame.
    """
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    return df


def explore_data(df):
    """
    Print basic EDA details: shape, head, missing values, data types.
    """
    print("\n===== Exploratory Data Analysis =====\n")

    # Dataset shape
    print("Dataset Shape:", df.shape)
    print("Rows   :", df.shape[0])
    print("Columns:", df.shape[1])

    # First 5 rows
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Basic statistics
    print("\n--- Rating Statistics ---")
    print(df["rating"].describe())
