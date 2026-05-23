import numpy as np
import pandas as pd
from src.logistic_regression import model

def accuracy(pred,true):
    return np.mean(pred==true)
def test_normalize(df,means,stds):
    df=df.copy()
    i=0
    for col in df.columns[:-1]:
        df[col] = (df[col]-means[i])/stds[i] # The normalization z= (x-mu)/sigma ~ N(0,1)
        i+=1
    return df
def test(df,weights,bias):
    X=df[df.columns[:-1]].to_numpy()
    Y=df[df.columns[-1]].to_numpy()
    predictions = model(weights,bias,X)
    predictions = (predictions>=0.5).astype(int)
    return accuracy(predictions,Y)
