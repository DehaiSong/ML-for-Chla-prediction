import torch
from torch import nn
import h5py
import time
import numpy as np
import os
import torch.utils.data as Data
from torch.utils.data import TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pytorchtools import EarlyStopping
import scipy.io as scio
import shap
import pandas as pd

# 导入数据
file1 = '/lustre/home/songdehai/lihai/matlab/cnn_5paras_new.mat'
file2 = '/lustre/home/songdehai/lihai/matlab/cnn_5paras_new_train.mat'
mat1 = h5py.File(file1,'r+')
mat2 = h5py.File(file2,'r+')
x_train=np.array(mat1['x_train']).T
y_train=np.array(mat1['y_train'])
x_valid=np.array(mat1['x_valid']).T
y_valid=np.array(mat1['y_valid'])
x_test2=np.array(mat2['PN_patch_train']).T
x_test=np.array(mat1['PN_patch_test']).T
print(x_test2.shape)
print(x_test.shape)






class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 20, 3, 1, 1),  # 10x21x21 => 20x21x21
            nn.Conv2d(20, 20, 3, 1, 1),  # 20x21x21 =>20x21x21
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 20x10x10 =>20x10x10
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, 3, 1, 1),  # 20x10x10 => 50x10x10
            nn.Conv2d(50, 50, 3, 1, 1),  # 50x10x10 => 50x10x10
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 50x10x10 => 50x5x5
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, 3, 1, 1),  # 50x5x5 => 100x5x5
            nn.Conv2d(100, 100, 3, 1, 1),  # 100x5x5 => 100x5x5
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 100x5x5 => 100x2x2
        )
        self.layer4 = nn.Sequential(
            nn.Linear(100 * 1 * 1, 200),  # 400 => 500
            nn.ReLU(),
            nn.Linear(200, 1),  # 500 => 1
        )

    # 前馈网络过程
    def forward(self, input):  # input=10x21x21
        input1 = self.layer1(input)  # 20x10x10
        input2 = self.layer2(input1)  # 50x5x5
        input3 = self.layer3(input2)  # 100x2x2
        x = input3.view(input3.size(0), -1)  # 400
        output = self.layer4(x)  # 1
        return output



# # 参数声明
epoches, batch_size, lr, workers, patience  = 100, 256, 0.001, 2, 5

train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))#list转tensor
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))#list转tensor
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=workers)


#损失函数和优化器
model = ConvNet()
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
        batch_x = batch_x
        batch_y = batch_y
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
        batch_x = batch_x
        batch_y = batch_y
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

# test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=workers)
# model.eval()
# test_loss= []
# for i, (batch_x, batch_y) in enumerate(test_loader):
#     batch_x = batch_x
#     batch_y = batch_y
#     output  = model(batch_x)
#     #output = torch.pow(10,output)
#     #batch_y = torch.pow(10,batch_y)
#     loss = torch.sqrt(criterion(output, batch_y))
#     test_loss.append(loss.item())
# test_loss_mean=np.mean(test_loss)
# print('Test : \t RMSE Loss: {:.8f}'.format(test_loss_mean))

x_test=torch.FloatTensor(x_test)
#y_test=torch.FloatTensor(y_test)
test_loss_month = []
cnn_chla1=torch.Tensor(x_test.size(0),x_test.shape[1])
for i in range(0,x_test.shape[1]):
    cnn_chla1[:,i]= torch.squeeze(model(torch.squeeze(x_test[:,i,:,:,:])))
    #output = torch.pow(10,output)
    #batch_y = torch.pow(10,batch_y)
    # loss = torch.sqrt(criterion(cnn_chla[:,i], y_test[:,i]))
    # test_loss_month.append(loss.item())
cnn_chla1=cnn_chla1.detach().numpy()
scio.savemat('cnn_chla_5paras_new.mat', {'cnn_chla1':cnn_chla1})


x_test2=torch.FloatTensor(x_test2)
cnn_chla2=torch.Tensor(x_test2.size(0),x_test2.shape[1])
for i in range(0,x_test2.shape[1]):
    cnn_chla2[:,i]= torch.squeeze(model(torch.squeeze(x_test2[:,i,:,:,:])))
    #output = torch.pow(10,output)
    #batch_y = torch.pow(10,batch_y)
    # loss = torch.sqrt(criterion(cnn_chla[:,i], y_test[:,i]))
    # test_loss_month.append(loss.item())
cnn_chla2=cnn_chla2.detach().numpy()
scio.savemat('cnn_chla_5paras_new_train.mat', {'cnn_chla2':cnn_chla2})

train_loss=np.array(train_loss)
valid_loss=np.array(valid_loss)
scio.savemat('train_loss.mat', {'train_loss':train_loss})
scio.savemat('valid_loss.mat', {'valid_loss':valid_loss})

# 创建一个绘图窗口
plt.figure()
epochs = range(0, len(train_loss))
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, valid_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Training and validation loss.jpg')