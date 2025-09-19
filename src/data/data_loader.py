import pandas as pd
import os
from pathlib import Path


def load_enphase_data(filename="4136754_custom_report.csv"):
    """
    Load Enphase solar data from CSV file

    Args:
        filename: Name of the CSV file in data/raw/

    Returns:
        pandas.DataFrame: Loaded solar data
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "raw" / filename

    # Load the data
    df = pd.read_csv(file_path)

    # Basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")

    return df


def basic_data_info(df):
    """Print basic information about the dataset"""
    print("\n=== Dataset Overview ===")
    print(df.info())
    print("\n=== First few rows ===")
    print(df.head())
    print("\n=== Missing values ===")
    print(df.isnull().sum())
