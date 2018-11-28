# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:22:11 2018

@author: Muhammad Dawood
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.preprocessing import StandardScaler

def xentropy_cost(x_target, x_pred):
 assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
 logged_x_pred = torch.log(x_pred)
 cost_value = -torch.sum(x_target * logged_x_pred)
 return cost_value


# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs+e) > 0).ravel()
    return (xs,x,y)

# experiment 2: noiseless inputs as privileged info (violates causal assump)
def synthetic_02(a,n):
    x  = np.random.randn(n,a.size)
    e  = np.random.randn(n,a.size)
    y  = (np.dot(x,a) > 0).ravel()
    xs = np.copy(x)
    x  = x+e
    return (xs,x,y)

# experiment 3: relevant inputs as privileged info
def synthetic_03(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:3]
    a  = a[0:3]
    y  = (np.dot(xs,a) > 0).ravel()
    return (xs,x,y)

# experiment 4: sample dependent relevant inputs as privileged info
def synthetic_04(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    #xs = np.sort(xs,axis=1)[:,::-1][:,0:3]
    xs = xs[:,np.random.permutation(a.size)[0:3]]
    a  = a[0:3]
    tt = np.median(np.dot(xs,a))
    y  = (np.dot(xs,a) > tt).ravel()
    return (xs,x,y)

    
def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

class Net(nn.Module):
    def __init__(self,d,q):
        super(Net, self).__init__()
#        q = 2
#        self.hidden1 = nn.Linear(d,1)
        self.out = nn.Linear(d,q)

    def forward(self,x):
#        x = self.hidden1(x)
        x = F.sigmoid(self.out(x))
        return x,x    


def fitModel(model,optimizer,criterion,epochs,x,target):
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = model(x)
        # Compute and print loss
        loss = criterion(y_pred, target)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
def do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te):
    t = 0
    l_r=0.001
    epochs=1000
    criterion = torch.nn.BCELoss()
    # scale stuff
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)
    xs_te = s_s.transform(xs_te)
    """
    Training of privilage space model
    """
    xs_tr = Variable(torch.from_numpy(xs_tr)).type(torch.FloatTensor)
    y_tr = Variable(torch.from_numpy(y_tr)).type(torch.FloatTensor)
    mlp_priv = Net(xs_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_priv.parameters())
    mlp_priv=fitModel(mlp_priv,optimizer,criterion,epochs,xs_tr,y_tr)   
    xs_te = Variable(torch.from_numpy(xs_te)).type(torch.FloatTensor)
    _,p_tr=mlp_priv(xs_tr)
    p_tr = p_tr.detach()
    output,_=mlp_priv(xs_te)
    pred = output > 0.5
    pred=pred.numpy().flatten()
    res_priv=np.mean(pred==y_te)      
    
    """
    Training of regular MLP
    """
    x_tr = Variable(torch.from_numpy(x_tr)).type(torch.FloatTensor)
    mlp_reg = Net(x_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_reg.parameters())
    mlp_reg=fitModel(mlp_reg,optimizer,criterion,epochs,x_tr,y_tr)
    x_te = Variable(torch.from_numpy(x_te)).type(torch.FloatTensor)
    output,_=mlp_reg(x_te)
    pred = output > 0.5
    pred=pred.numpy().flatten()
    res_reg=np.mean(pred==y_te)

#    softened=soften.detach()
#    p_tr=softened.numpy()
#    import pdb; pdb.set_trace()
#    #p_tr=softmax(softened,t)
#    p_tr=Variable(torch.from_numpy(p_tr)).type(torch.FloatTensor)
    
    ### freezing layers
    for param in mlp_priv.parameters():
        param.requires_grad =False
    
    """
    LUPI Combination of two model
    """
    mlp_dist = Net(x_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_dist.parameters())
    criterion = torch.nn.BCELoss()
    # Training loop
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = mlp_dist(x_tr)
        # Compute and print loss
#        loss1 = (1-l)*criterion(y_pred, y_tr)
#        loss2=t*t*l*criterion(y_pred, p_tr)
#        loss=loss1+loss2
#        loss =l*(torch.exp(-t*criterion(p_tr , y_tr))*criterion(y_pred , p_tr))  +    (1-l)*criterion(y_pred,y_tr)
        loss = criterion(y_pred,y_tr) + torch.exp(-t*criterion(p_tr,y_tr))*(criterion(y_pred,p_tr) - criterion(y_pred,y_tr))
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output,_=mlp_dist(x_te)
    pred = output > 0.5
    pred=pred.numpy().flatten()
    res_dis=np.mean(pred==y_te)
    return np.array([res_priv,res_reg,res_dis])


# experiment hyper-parameters
d      = 50
n_tr   = 200
n_te   = 1000
n_epochs = 10
eid    = 0

np.random.seed(0)

# do all four experiments
print("\nDistillation Pytorch Regression  Using BCE Loss method")
for experiment in (synthetic_01,synthetic_02,synthetic_03,synthetic_04):
    eid += 1
    R = np.zeros((n_epochs,3))
    for rep in range(n_epochs):
        a   = np.random.randn(d)
        (xs_tr,x_tr,y_tr) = experiment(a,n=n_tr)
#        1/0
        (xs_te,x_te,y_te) = experiment(a,n=n_te)
        R[rep,:] += do_exp(x_tr,xs_tr,y_tr*1.0,x_te,xs_te,y_te*1.0)
    means = R.mean(axis=0).round(2)
    stds  = R.std(axis=0).round(2)
    print (str(eid)+\
          ' # '+str(means[0])+'+/-'+str(stds[0])+\
          ' # '+str(means[1])+'+/-'+str(stds[1])+\
          ' # '+str(means[2])+'+/-'+str(stds[2]))