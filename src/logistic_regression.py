import pandas as pd
import numpy as np

def bce(pred,y):
    '''
    BCE is y_i*log(y_pred)+(1-y_i)*log(1-y_pred)
    y_pred here is predicted label model(x)
    y_i here is true label y
    '''
    pred = np.clip(pred,1e-9,1-1e-9)
    return -(y*np.log(pred)+(1-y)*np.log(1-pred))

def elastic_net(weights, lambda1,lambda2):
    return lambda1*np.sum(np.abs(weights))+lambda2*np.sum(np.square(weights))

def model(weights,bias,X):
    z=X@weights+bias
    return 1/(1+np.exp(-z))

def loss(target,predictions,weights,lambda1,lambda2):
    loss=np.mean(bce(predictions,target))
    loss+=elastic_net(weights,lambda1,lambda2)
    return loss

def grad_desc(lr,X,predictions,target,weights,bias,lambda1,lambda2):
    w_update = ((predictions-target)@X)/len(predictions)
    reg_update = lambda1*np.sign(weights) + 2*lambda2*weights
    b_update = np.sum(predictions-target)/len(predictions)
    weights = weights - lr*(w_update+reg_update)
    bias = bias - lr*b_update
    return weights,bias

def train(epochs,df,lr,lambda1,lambda2):
    weights=np.zeros(len(df.columns)-1)
    bias=0.0
    X=df[df.columns[:-1]].to_numpy()
    Y=df[df.columns[-1]].to_numpy()
    for epoch in range(epochs):
        print(f'Running epoch {epoch}')
        predictions = model(weights,bias,X)
        weights,bias = grad_desc(lr,X,predictions,Y,weights,bias,lambda1,lambda2)
        print(f'loss incurred {loss(Y,predictions,weights,lambda1,lambda2)}')
        predictions = (predictions>=0.5).astype(int)
        accuracy =np.mean(predictions==Y)
        print(f'accuracy is {accuracy}')
    return weights,bias