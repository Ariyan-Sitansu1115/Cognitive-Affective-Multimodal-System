import os

import pandas as pd


def load_raw_data(csv_files):
    assert len(csv_files) >= 1, f"No CSV found among: {csv_files[:15]}"
    raw_file = max(csv_files, key=os.path.getsize)
    print("Using raw file:", raw_file)
    dataframe = pd.read_csv(raw_file)
    print(dataframe.shape)
    print(list(dataframe.columns))
    print(dataframe.head())
    print(dataframe.columns.tolist())
    return raw_file, dataframe
