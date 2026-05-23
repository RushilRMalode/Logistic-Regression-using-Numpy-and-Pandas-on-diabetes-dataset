import pandas as pd
import numpy as np
# need to perform z-score normalization on everything except the target variable

def z_score_normalization(df: pd.DataFrame) -> pd.DataFrame: # This is called typehints, interesting
    df=df.copy()
    for col in df.columns[:-1]:
        mean = np.mean(df[col])
        std = np.std(df[col])
        df[col] = (df[col]-mean)/std # The normalization z= (x-mu)/sigma ~ N(0,1)
    return df

def train_test_split(df: pd.DataFrame,split_ratio: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    sample_size = int(len(df)*split_ratio)
    train_idx = np.random.choice(len(df),sample_size,replace=False)
    train = df.iloc[train_idx]
    test_idx = np.setdiff1d(np.arange(len(df)),train_idx)
    test = df.iloc[test_idx]
    return train,test