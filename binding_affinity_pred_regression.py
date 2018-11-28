# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:35:14 2018

@author: Wajid Abbasi
"""
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error,roc_auc_score,average_precision_score
from sklearn.model_selection import LeaveOneOut
from sklearn import cross_validation
from torch.autograd import Variable
import scipy

def load_affinity_values(info_path, column):
    "This method return experimental affinity values as dictionary with complex ID as key and affinity as value"
    affinity_data_dic={}
    affinity_data=np.loadtxt(info_path,dtype='str',delimiter='\t')#[0:]
    affinity_data=affinity_data[1:]
    for i in range(len(affinity_data)):
        if affinity_data[i][0][:4] in affinity_data_dic.keys():
            print (affinity_data[i][0][:4])
        affinity_data_dic[affinity_data[i][0][:4]]=[float(affinity_data[i][column]),affinity_data[i][1]]
    return affinity_data_dic
    
def mean_varrianace_normalization(examples,std='T'):
    from sklearn import preprocessing
    #np.save('2mer_ungoup_feats_mean.npy',np.mean(examples,axis=0))
    #np.save('2mer_ungoup_feats_std.npy',((np.std(examples,axis=0))+1e-9))
    if std=='T':
        examples=preprocessing.scale(examples)
    examples=preprocessing.normalize(examples)
    return examples
    
    
def generate_standardized_complex_level_examples(feat_path,affinity_info,normalize='T'):
    values=[]
    complex_id=[]
    examples=[]
    group=[]
    #affinity_info=load_affinity_values('/media/wajid/PhD_Projects/Projects/Affinity_prediction_project/dataset/yugandhar_info.txt', affinity_col)
    for key in affinity_info:
        if os.path.exists(feat_path+key+'.npy') and os.path.exists('features/complex_level/Dias_etal_2017/'+key+'.npy')and os.path.exists('features/complex_level/moal_2011/'+key+'.npy'):
            examples.append(np.load(feat_path+key+'.npy'))
            values.append(affinity_info[key][0])
            complex_id.append(key)
            group.append(affinity_info[key][1])
        else:
            print (key)
    if normalize=='T':
        examples=mean_varrianace_normalization(np.asarray(examples))
    return np.asarray(examples),np.asarray(values),np.asarray(complex_id),np.asarray(group)

    
def generate_standardized_examples(feat_path,affinity_info,normalize='T'):
    values=[]
    complex_id=[]
    ligand=[]
    receptor=[]
    group=[]
    #affinity_info=load_affinity_values('/media/wajid/PhD_Projects/Projects/Affinity_prediction_project/LUPI/dataset_for_clf/yugandhar_info.txt', affinity_col)
    for key in affinity_info:
        if os.path.exists(feat_path+key+'_l_u.npy') and os.path.exists(feat_path+key+'_r_u.npy') and os.path.exists('features/complex_level/Dias_etal_2017/'+key+'.npy')and os.path.exists('features/complex_level/moal_2011/'+key+'.npy'):
            ligand.append(np.load(feat_path+key+'_l_u.npy'))
            receptor.append(np.load(feat_path+key+'_r_u.npy'))
            values.append(affinity_info[key][0])
            complex_id.append(key)
            group.append(affinity_info[key][1])
    if normalize=='T':    
        #ligand=mean_varrianace_normalization(np.asarray(ligand),std='F')
        #receptor==mean_varrianace_normalization(np.asarray(receptor),std='F')
        examples=[np.concatenate((ligand[i],receptor[i]),axis=0) for i in range(len(ligand))]
        examples=mean_varrianace_normalization(np.asarray(examples))
 
    else:
        examples=[np.concatenate((ligand[i],receptor[i]),axis=0) for i in range(len(ligand))]
        #examples=mean_varrianace_normalization(np.asarray(examples))
    return np.asarray(examples),np.asarray(values),np.asarray(complex_id),np.asarray(group)    

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]


def fitModel(model,optimizer,criterion,epochs,x,target):
    for epoch in range(epochs):
#        import pdb; pdb.set_trace()
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = model(x)
        # Compute and print loss
        loss = criterion(y_pred, target)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return model
    


def train_test_clf(examples,labels,aff,complex_id,group,epochs=100,soft_labels=None,dist='no',t=5000,l=0.5,act_labels=None,l_r=0.06):
    predicted_score=[]
    actual_score=[]
    loo = cross_validation.LeaveOneOut(examples.shape[0])
    if os.path.exists('out_file.txt'):
        os.remove('out_file.txt')
    for train_i, test_i in loo:
        X=Variable(torch.from_numpy(examples[train_i])).type(torch.FloatTensor)
        Y=Variable(torch.from_numpy(labels[train_i])).type(torch.FloatTensor)
        Xv=Variable(torch.from_numpy(examples[test_i])).type(torch.FloatTensor)
        if dist=='yes':
            model=Net(X.shape[1],1)
            optimizer = optim.SGD(model.parameters(),lr=l_r)
            criterion = torch.nn.MSELoss()
            Ys=Variable(torch.from_numpy(soft_labels[train_i])).type(torch.FloatTensor)
            # Training loop
            for epoch in range(epochs):
                    # Forward pass: Compute predicted y by passing x to the model
                y_pred,_ = model(X)
#                criterion(y_pred,y_tr) + torch.exp(-t*criterion(p_tr,y_tr))*(criterion(y_pred,p_tr) - criterion(y_pred,y_tr))
                loss = criterion(y_pred,Y) + torch.exp(-t*criterion(Ys,Y))*(criterion(y_pred,Ys) - criterion(y_pred,Y))
#                loss =l*(torch.exp(-t*criterion(Ys , Y))*criterion(y_pred , Ys))  +    (1-l)*criterion(y_pred,Y)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        else: 
            model=Net(X.shape[1],1)
#            import pdb;pdb.set_trace()
            model=fitModel(model,optim.SGD(model.parameters(),lr=l_r),torch.nn.MSELoss(),epochs,X,Y)
        #print svr.best_estimator_
        model.eval()
        output,_=model(Xv)
        output=output[0].detach().numpy()
#        print(complex_id[test_i],group[test_i],aff[test_i],output)
        predicted_score.append(output[0])
        actual_score.append(aff[test_i][0])
    pred=np.array(predicted_score)
    act=np.array(actual_score)
    pred=pred.reshape(pred.shape[0])
    act=act.reshape(act.shape[0])
    print ("RMSE:",np.sqrt(mean_squared_error(act,pred)))
    print("Correlation coeffience",np.corrcoef(pred,act)[1,0])
    print ("Pearson Correlation Coefficient is:",scipy.stats.pearsonr(pred,act))
    print ("Spearman Correlation Coefficient is:",scipy.stats.spearmanr(pred,act))
    print ("AUC-ROC:",roc_auc_score(act_labels,pred))
    print ("AUC-PR:",average_precision_score(act_labels,pred))
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(pred, act, "b*")
    return model
    
    

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math

class Net(nn.Module):
    def __init__(self,d,q):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,math.floor(d/3))
        self.hidden2 = nn.Linear(math.floor(d/3),q)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.hidden2(x)
#        x1 = F.softmax(x,dim=1)
        return x,x

affinity_info=load_affinity_values('yugandhar_info.txt', 2)
previlleged_feats,aff_values,complex_id,group=generate_standardized_complex_level_examples('./features/complex_level/moal_2011/',affinity_info,normalize='F')
previlleged_feats=previlleged_feats[:,1:]
previlleged_feats=mean_varrianace_normalization(np.asarray(previlleged_feats))
act_labels=np.asarray([1]*len(aff_values))
act_labels[np.where( aff_values<-10.86)]=0

feats,aff_values,complex_id,group=generate_standardized_examples('features/2-mer/',affinity_info,normalize='T')    
aff_values=aff_values.reshape(aff_values.shape[0],1)
labels=aff_values
examples=previlleged_feats
print("Regular model")
lr=np.linspace(0,1,10)
train_test_clf(feats,labels,aff_values,complex_id,group,epochs=100,act_labels=act_labels,l_r=0.06)
print("Previlleged space model")
previlleged_feats=np.hstack((feats,previlleged_feats))
mlp_priv=train_test_clf(previlleged_feats,labels,aff_values,complex_id,group,epochs=50,act_labels=act_labels,l_r=0.06)
_,soften=mlp_priv(Variable(torch.from_numpy(previlleged_feats)).type(torch.FloatTensor))
softened=soften.detach()
p_tr=softened.numpy()
print("Distillation based model")
train_test_clf(feats,labels,aff_values,complex_id,group,epochs=100,dist='yes',soft_labels=p_tr,act_labels=act_labels,l_r=0.06,t=0.0,l=0.5)