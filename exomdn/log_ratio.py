import numpy as np
import pandas as pd


def transform_df(data: pd.DataFrame, base: str,
                 columns: list[str], new_columns: list[str]) -> pd.DataFrame:
    for i, x in enumerate(columns):
        col_name = new_columns[i]
        data[col_name] = np.log((data[x]) / (data[base]))
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=new_columns, how="any")


def inv_transform_df(data: pd.DataFrame, base: str,
                     columns: list[str], new_columns: list[str]) -> pd.DataFrame:
    for i, x in enumerate(columns):
        col_name = new_columns[i]
        data[col_name] = np.exp(data[x]) / (np.exp(data[columns]).sum(axis=1) + 1)
    data[base] = 1 - data[new_columns].sum(axis=1)
    return data
