import scipy.io as scio
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
import os
from torch.utils.data import TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
from pytorchtools import EarlyStopping

# 导入数据
file='forest_data_5paras.mat'
mat = h5py.File(file,'r+')
chla_train_forest=np.array(mat['chla_train_forest']).T
PN_train_forest=np.array(mat['PN_train_forest']).T
chla_test_forest=np.array(mat['chla_test_forest']).T
PN_test_forest=np.array(mat['PN_test_forest']).T
PN_test_forest2=np.array(mat['PN_test_forest2']).T

print(PN_train_forest.shape)
print(chla_train_forest.shape)
print(PN_test_forest2.shape)
print(PN_test_forest.shape)

#数据归一化
scaler = MinMaxScaler()
x_train_scaled = scaler.fit(PN_train_forest).transform(PN_train_forest)
#x_valid_scaled = scaler.fit(x_train).transform(x_valid)
x_test_scaled = scaler.fit(PN_train_forest).transform(PN_test_forest)

x_train, x_valid, y_train, y_valid = train_test_split(x_train_scaled, chla_train_forest, test_size=0.3,random_state=20)

epoches, batch_size, lr, workers, patience = 100, 256, 0.001, 2, 5

train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))#list转tensor
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))#list转tensor
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=workers)

class Net(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
#         self.hidden3 = nn.Linear(n_hidden2,n_hidden3)
#         self.hidden4 = nn.Linear(n_hidden3,n_hidden4)
#         self.hidden5 = nn.Linear(n_hidden4,n_hidden5)
        self.predict = nn.Linear(n_hidden2,n_output)
        self.BN1=nn.BatchNorm1d(n_hidden1)
        self.BN2=nn.BatchNorm1d(n_hidden2)
#         self.BN3=nn.BatchNorm1d(n_hidden3)
#         self.BN4=nn.BatchNorm1d(n_hidden4)
#         self.BN5=nn.BatchNorm1d(n_hidden5)
    def forward(self,input):
        out = self.hidden1(input)
        out = self.BN1(out)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = self.BN2(out)
        out = torch.relu(out)
#         out = self.hidden2(out)
#         out = self.BN2(out)
#         out = torch.relu(out)
#         out=self.hidden3(out)
#         out = self.BN3(out)
#         out = torch.relu(out)
#         out=self.hidden4(out)
#         out = self.BN4(out)
#         out = torch.relu(out)
#         out=self.hidden5(out)
#         out = self.BN5(out)
#         out = torch.relu(out)
        out = self.predict(out)

        return out

model = Net(5,32,64,1)
print(model)

#损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_loss, valid_loss = [], []

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True)
for epoch in range(0, epoches):
    print("进行第{}个epoch".format(epoch))
    ###################
    # train the model #
    ###################
    model.train()
    total_train_loss = []
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch_x)
        # output = torch.pow(10,output)
        # batch_y = torch.pow(10,batch_y)
        loss = torch.sqrt(criterion(output, batch_y))
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    train_loss_mean = np.mean(total_train_loss)
    train_loss.append(train_loss_mean)
    print('Train Epoch: {}\t RMSE Loss: {:.6f}'.format(epoch, train_loss_mean))

    ######################
    # validate the model #
    ######################
    model.eval()
    total_valid_loss = []
    for i, (batch_x, batch_y) in enumerate(valid_loader):
        output = model(batch_x)
        # output = torch.pow(10,output)
        # batch_y = torch.pow(10,batch_y)
        loss = torch.sqrt(criterion(output, batch_y))
        total_valid_loss.append(loss.item())
    valid_loss_mean = np.mean(total_valid_loss)
    valid_loss.append(valid_loss_mean)
    print('Valid Epoch: {}\t RMSE Loss: {:.8f}'.format(epoch, valid_loss_mean))

    early_stopping(valid_loss_mean, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
model.load_state_dict(torch.load('checkpoint.pt'))

x_test=PN_test_forest2
x_test=torch.FloatTensor(x_test)
x_train=torch.FloatTensor(x_train)
x_scaled = torch.Tensor(x_test.shape[0],x_test.shape[1],x_test.shape[2])
pre_Chla_ann=torch.Tensor(x_test.size(0),x_test.shape[1])
for i in range(0,x_test.shape[1]):
    x_scaled[:, i, :] = scaler.fit(x_train).transform(x_test[:, i, :])
    pre_Chla_ann[:,i]= torch.squeeze(model(torch.squeeze(x_scaled[:,i,:])))
    #output = torch.pow(10,output)
    #batch_y = torch.pow(10,batch_y)
pre_Chla_ann=pre_Chla_ann.detach().numpy()
scio.savemat('pre_Chla_ann_nor.mat', {'pre_Chla_ann':pre_Chla_ann})