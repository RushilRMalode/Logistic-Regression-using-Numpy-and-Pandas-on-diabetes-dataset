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