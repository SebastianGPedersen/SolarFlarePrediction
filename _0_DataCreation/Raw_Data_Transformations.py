import numpy as np
import pandas as pd
from typing import List


def data_to_keep(df, nrow: int, cols_to_keep: List=range(16)):
    """
    Removes nrow from df, and keeps cols_to_keep (?not in place?)
    :param df:
    :param nrow:
    :param cols_to_keep:
    :return: (As copy) extract from df
    """
    return df.iloc[-nrow:, cols_to_keep]


def df_to_row(df):
    """
    Convert df.shape (x, y) to shape (1, x*y), append row index to original column names
    :param df:
    :return: reshaped dataframe
    """
    v = df.unstack().to_frame().sort_index(level=1).T
    v.columns = [t[0] + '_' + str(t[1]) for t in v.columns.values]
    return v


def scale_df(df: pd.DataFrame):
    """
    Scales columns in dataframe (in place)
    :param df: Dataframe to scale
    :return: The scaled df
    """
    for col in df.columns:
        if col in ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "XR_MAX"]:
            df[col] = scale_series(df[col], lambda x: np.sign(x) * np.abs(x) ** 0.2)
        if col in ["MEANPOT"]:
            df[col] = scale_series(df[col], lambda x: np.sign(x) * np.abs(x) ** 0.1)
        if col in ["TOTFZ", "TOTFY", "TOTFX"]:
            df[col] = scale_series(df[col], lambda x: np.sign(x) * np.abs(x) ** 0.5)

    return df


def scale_series(series, f):
    return np.fromiter((f(x) for x in series), series.dtype)

# [sign(x) * abs(x)^(1/5) for x in  ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "XR_MAX"]]
# [sign(x) * abs(x)^(1/10) for x in  ["MEANPOT"]]
# [sign(x) * abs(x)^(1/2) for x in  ["TOTFZ", "TOTFY", "TOTFX"]]


if __name__ == '__main__':
    from _0_DataCreation.Read_Data import load_data, fn
    id, label, df = next(load_data(filename=fn, max_row=1))
    df2 = scale_df(df)
    print(df)
    print(all(df == df2))
