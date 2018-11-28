#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:28:42 2018

@author: dslab01
"""
from keras.datasets import mnist
import numpy as np
from scipy.misc import imresize
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.preprocessing import StandardScaler

def load_data(n_sample=500,dim=7):
    """
    n_sample: the number of sample used in training
    dim is the downsample size
    """
    #np.random.seed(0)
    (x_train, y_train), (xs_test, y_test) = mnist.load_data()
    i     = np.random.permutation(x_train.shape[0])[0:n_sample]
    xs_train = x_train[i,]
    y_train = y_train[i,]
    x_train = np.zeros((xs_train.shape[0],dim,dim))
    x_test = np.zeros((xs_test.shape[0],dim,dim))
    for i in range(x_train.shape[0]):
        x_train[i,] = imresize(xs_train[i,],(dim,dim))
    for i in range(xs_test.shape[0]):
        x_test[i,] = imresize(xs_test[i,],(dim,dim))
    return xs_train/255.0,x_train/255.0,y_train,x_test/255.0,xs_test/255.0,y_test      


class Net(nn.Module):
    def __init__(self,d,q):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,16)
        self.hidden2 = nn.Linear(16,32)
        self.dp1     = nn.Dropout2d(p=0.2)
        self.dp2     = nn.Dropout2d(p=0.2)
        self.out = nn.Linear(32,q)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.dp1(F.relu(self.hidden1(x)))
        x = self.dp2(F.relu(self.hidden2(x)))
        x = F.softmax(self.out(x))
        return x
    
def fitModel(model,optimizer,criterion,epochs,x,target):
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, target)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model



epochs=50
criterion = torch.nn.CrossEntropyLoss()

priv=[]
reg=[]
dist=[]
t=0.6
l=1

for rep in range(10):
    
    xs_tr,x_tr,y_tr,x_te,xs_te,y_te = load_data()
    xs_tr = Variable(torch.from_numpy(xs_tr)).type(torch.FloatTensor)
    y_tr = Variable(torch.from_numpy(y_tr)).long()
    xs_te=np.reshape(xs_te,(xs_te.shape[0],784))
    xs_te = Variable(torch.from_numpy(xs_te)).type(torch.FloatTensor)
    
    x_tr = Variable(torch.from_numpy(x_tr)).type(torch.FloatTensor)
    x_te=np.reshape(x_te,(x_te.shape[0],49))
    x_te = Variable(torch.from_numpy(x_te)).type(torch.FloatTensor)

    

    """
    Training of privilag space model
    """
    mlp_priv = Net(784,10)
    optimizer = optim.RMSprop(mlp_priv.parameters())
    mlp_priv=fitModel(mlp_priv,optimizer,criterion,epochs,xs_tr,y_tr) 
    mlp_priv.eval()
    output=mlp_priv(xs_te)
    p_tr = mlp_priv(xs_tr)
    pred = torch.argmax(output,1)
    pred=pred.numpy()
    res_priv=np.mean(pred==y_te) 
    """
    Training of Input space model
    """
    mlp_reg = Net(49,10)
    optimizer = optim.RMSprop(mlp_reg.parameters())
    mlp_reg=fitModel(mlp_reg,optimizer,criterion,epochs,x_tr,y_tr) 
    mlp_reg.eval()
    output=mlp_reg(x_te)
    pred = torch.argmax(output,1)
    pred = pred.numpy()
    res_reg=np.mean(pred==y_te)  
    reg.append(res_reg)
    priv.append(res_priv)
    
    ### freezing layers
    for param in mlp_priv.parameters():
        param.requires_grad =False
    
    """
    LUPI Combination of two model
    """
    mlp_dist = Net(49,10)
    # best is 0.02 learning rate
    optimizer = optim.RMSprop(mlp_dist.parameters())
    # Training loop
    for epoch in range(700):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred= mlp_dist(x_tr)
        # Compute and print loss
#        loss1 = (1-l)*criterion(y_pred, y_tr)
#        loss2=t*t*l*criterion(y_pred, p_tr)
#        loss=loss1+loss2
        loss = criterion(y_pred,y_tr) + torch.exp(-t*criterion(p_tr,y_tr))*(criterion(y_pred,torch.argmax(p_tr,1).long()) - criterion(y_pred,y_tr))
        #loss =l*(torch.exp(-t*criterion(p_tr , y_tr))*criterion(y_pred ,torch.argmax(p_tr,1).long()))  +    (1-l)*criterion(y_pred,y_tr)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    mlp_dist.eval()
    output=mlp_dist(x_te)
    pred = torch.argmax(output,dim=1)
    pred = pred.numpy()
    res_dis=np.mean(pred==y_te)
    dist.append(res_dis)
    print(res_priv,res_reg,res_dis)
#pred = output > 0.5
print(np.mean(priv),np.mean(reg),np.mean(dist))


