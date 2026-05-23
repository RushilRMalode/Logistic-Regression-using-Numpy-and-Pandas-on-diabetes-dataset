import pandas as pd
import numpy as np
from src.preprocessing import z_score_normalization,train_test_split
from src.logistic_regression import train
from src.testing import test,test_normalize

df = pd.read_csv('data/diabetes[1].csv')
train_data,test_data=train_test_split(df,0.8)
train_data,means,stds=z_score_normalization(train_data)
test_data = test_normalize(test_data,means,stds)
weights,bias=train(30,train_data,0.01,0.01,0.01)
print(test(test_data,weights,bias))

