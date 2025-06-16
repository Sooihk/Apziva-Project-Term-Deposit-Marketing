import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_deposit():
    # Get path relative to this file's location
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / "raw" / "term-deposit-marketing-2020.csv"
    term_deposit = pd.read_csv(data_path)
    return term_deposit

if __name__ == "__main__":
    df = load_deposit()
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("Preview:")
    print(df.info())
    