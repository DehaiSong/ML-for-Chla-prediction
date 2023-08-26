import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch.utils.data import TensorDataset
from pytorchtools import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# Hyper Parameters
TIME_STEP = 36    # rnn time step
INPUT_SIZE = 1     # rnn input size
LR = 0.01           # learning rate
epoches = 100
workers = 2
patience = 5
batch_size=256


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX),np.array(dataY)

# 导入数据
file1='lstm_input(month_ts36)_new2.mat'
mat1 = h5py.File(file1,'r+')
x=np.array(mat1['x_train']).T
y=np.array(mat1['y_train']).T
x_test=np.array(mat1['x_test']).T
y_test=np.array(mat1['y_test']).T
# x_train1, x_valid1, y_train1, y_valid1 = train_test_split(x, y, test_size=0.9,random_state=40)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.3,random_state=50)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3,random_state=20)


print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)

train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))#list转tensor
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))#list转tensor
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=workers)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, (h_state, cell) = self.rnn(x, None)
        #         r_out = r_out[:,-1]
        #         outs = self.out(r_out)
        #         return outs[-1]

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        outs = self.out(r_out)
        return outs


model = RNN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
criterion = nn.MSELoss()

# train_loss, valid_loss = [], []
#
# # initialize the early_stopping object
# early_stopping = EarlyStopping(patience=patience, verbose=True)
# for epoch in range(0, epoches):
#     print("进行第{}个epoch".format(epoch))
#     ###################
#     # train the model #
#     ###################
#     model.train()
#     total_train_loss = []
#     for i, (batch_x, batch_y) in enumerate(train_loader):
#         optimizer.zero_grad()
#         batch_x=batch_x.unsqueeze(dim=2)
#         output = model(batch_x)
#         # output = torch.pow(10,output)
#         # batch_y = torch.pow(10,batch_y)
#         loss = torch.sqrt(criterion(output[:,-1,:], batch_y))
#         loss.backward()
#         optimizer.step()
#         total_train_loss.append(loss.item())
#     train_loss_mean = np.mean(total_train_loss)
#     train_loss.append(train_loss_mean)
#     print('Train Epoch: {}\t RMSE Loss: {:.6f}'.format(epoch, train_loss_mean))
#
#     ######################
#     # validate the model #
#     ######################
#     model.eval()
#     total_valid_loss = []
#     for i, (batch_x, batch_y) in enumerate(valid_loader):
#         batch_x=batch_x.unsqueeze(dim=2)
#         output = model(batch_x)
#         # output = torch.pow(10,output)
#         # batch_y = torch.pow(10,batch_y)
#         loss = torch.sqrt(criterion(output[:,-1,:], batch_y))
#         total_valid_loss.append(loss.item())
#     valid_loss_mean = np.mean(total_valid_loss)
#     valid_loss.append(valid_loss_mean)
#     print('Valid Epoch: {}\t RMSE Loss: {:.8f}'.format(epoch, valid_loss_mean))
#
#     early_stopping(valid_loss_mean, model)
#     if early_stopping.early_stop:
#         print("Early stopping")
#         break
model.load_state_dict(torch.load('checkpoint.pt'))

# train_loss=np.array(train_loss)
# valid_loss=np.array(valid_loss)
# scio.savemat('train_loss.mat', {'train_loss':train_loss})
# scio.savemat('valid_loss.mat', {'valid_loss':valid_loss})
#
# x_valid=torch.FloatTensor(x_valid)
# valid_true=model(x_valid.unsqueeze(dim=2))[:,-1,:]
# valid_true=valid_true.detach().numpy()
# scio.savemat('y_valid.mat', {'y_valid':y_valid})
# scio.savemat('valid_true.mat', {'valid_true':valid_true})
#
#
#
# #model.load_state_dict(torch.load('checkpoint_rnn_new2.pt', map_location=torch.device('cpu')))
# #单步逐日预测
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test)
# #out=torch.Tensor(y_test.size(0),y_test.shape[3])
# rnn_chla=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in range(0,y_test.shape[1]):
#     print(i)
#     rnn_chla[:,i]= torch.squeeze(model(x_test[:,:,i].unsqueeze(dim=2))[:,-1,:])
#     #output = torch.pow(10,output)
#     #batch_y = torch.pow(10,batch_y)
#     # loss = torch.sqrt(criterion(cnn_chla[:,i], y_test[:,i]))
#     # test_loss_month.append(loss.item())
# rnn_chla=rnn_chla.detach().numpy()
# scio.savemat('rnn_chla.mat', {'rnn_chla':rnn_chla})
#
#
# #单步滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in range(0,y_test.shape[1]):
#     print(i)
#     rnn_chla_roll[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#     a=torch.cat((a[:,1:],rnn_chla_roll[:,i].unsqueeze(dim=1)),1)
#     #output = torch.pow(10,output)
#     #batch_y = torch.pow(10,batch_y)
#     # loss = torch.sqrt(criterion(cnn_chla[:,i], y_test[:,i]))
#     # test_loss_month.append(loss.item())
# rnn_chla_roll=rnn_chla_roll.detach().numpy()
# scio.savemat('rnn_chla_roll.mat', {'rnn_chla_roll':rnn_chla_roll})
#
# #单步容忍滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll_1=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in np.arange(y_test.shape[1]):
#     print(i)
#     if i <= 0:
#         rnn_chla_roll_1[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#         a=torch.cat((a[:,1:],rnn_chla_roll_1[:,i].unsqueeze(dim=1)),1)
#     if i > 0:
#         p=torch.cat((x_test[:,0:35,i],a[:,-1:]),1)
#         rnn_chla_roll_1[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
#         a = torch.cat((a[:, 1:], rnn_chla_roll_1[:, i].unsqueeze(dim=1)), 1)
# rnn_chla_roll_1=rnn_chla_roll_1.detach().numpy()
# scio.savemat('rnn_chla_roll_1.mat', {'rnn_chla_roll_1':rnn_chla_roll_1})
#
# #单步容忍滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll_5=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in np.arange(y_test.shape[1]):
#     print(i)
#     if i <= 4:
#         rnn_chla_roll_5[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#         a=torch.cat((a[:,1:],rnn_chla_roll_5[:,i].unsqueeze(dim=1)),1)
#     if i > 4:
#         p=torch.cat((x_test[:,0:31,i],a[:,-5:]),1)
#         rnn_chla_roll_5[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
#         a = torch.cat((a[:, 1:], rnn_chla_roll_5[:, i].unsqueeze(dim=1)), 1)
# rnn_chla_roll_5=rnn_chla_roll_5.detach().numpy()
# scio.savemat('rnn_chla_roll_5.mat', {'rnn_chla_roll_5':rnn_chla_roll_5})
#
# #单步容忍滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll_10=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in np.arange(y_test.shape[1]):
#     print(i)
#     if i <= 9:
#         rnn_chla_roll_10[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#         a=torch.cat((a[:,1:],rnn_chla_roll_10[:,i].unsqueeze(dim=1)),1)
#     if i > 9:
#         p=torch.cat((x_test[:,0:26,i],a[:,-10:]),1)
#         rnn_chla_roll_10[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
#         a = torch.cat((a[:, 1:], rnn_chla_roll_10[:, i].unsqueeze(dim=1)), 1)
# rnn_chla_roll_10=rnn_chla_roll_10.detach().numpy()
# scio.savemat('rnn_chla_roll_10.mat', {'rnn_chla_roll_10':rnn_chla_roll_10})
#
# #单步容忍滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll_15=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in np.arange(y_test.shape[1]):
#     print(i)
#     if i <= 14:
#         rnn_chla_roll_15[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#         a=torch.cat((a[:,1:],rnn_chla_roll_15[:,i].unsqueeze(dim=1)),1)
#     if i > 14:
#         p=torch.cat((x_test[:,0:21,i],a[:,-15:]),1)
#         rnn_chla_roll_15[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
#         a = torch.cat((a[:, 1:], rnn_chla_roll_15[:, i].unsqueeze(dim=1)), 1)
# rnn_chla_roll_15=rnn_chla_roll_15.detach().numpy()
# scio.savemat('rnn_chla_roll_15.mat', {'rnn_chla_roll_15':rnn_chla_roll_15})
#
# #单步容忍滚动预测
# a=x_test[:,:,0]
# rnn_chla_roll_20=torch.Tensor(y_test.shape[0],y_test.shape[1])
# for i in np.arange(y_test.shape[1]):
#     print(i)
#     if i <= 19:
#         rnn_chla_roll_20[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
#         a=torch.cat((a[:,1:],rnn_chla_roll_20[:,i].unsqueeze(dim=1)),1)
#     if i > 19:
#         p=torch.cat((x_test[:,0:16,i],a[:,-20:]),1)
#         rnn_chla_roll_20[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
#         a = torch.cat((a[:, 1:], rnn_chla_roll_20[:, i].unsqueeze(dim=1)), 1)
# rnn_chla_roll_20=rnn_chla_roll_20.detach().numpy()
# scio.savemat('rnn_chla_roll_20.mat', {'rnn_chla_roll_20':rnn_chla_roll_20})

#单步容忍滚动预测
a=x_test[:,:,0]
rnn_chla_roll_25=torch.Tensor(y_test.shape[0],y_test.shape[1])
for i in np.arange(y_test.shape[1]):
    print(i)
    if i <= 24:
        rnn_chla_roll_25[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
        a=torch.cat((a[:,1:],rnn_chla_roll_25[:,i].unsqueeze(dim=1)),1)
    if i > 24:
        p=torch.cat((x_test[:,0:11,i],a[:,-25:]),1)
        rnn_chla_roll_25[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
        a = torch.cat((a[:, 1:], rnn_chla_roll_25[:, i].unsqueeze(dim=1)), 1)
rnn_chla_roll_25=rnn_chla_roll_25.detach().numpy()
scio.savemat('rnn_chla_roll_25.mat', {'rnn_chla_roll_25':rnn_chla_roll_25})

#单步容忍滚动预测
a=x_test[:,:,0]
rnn_chla_roll_30=torch.Tensor(y_test.shape[0],y_test.shape[1])
for i in np.arange(y_test.shape[1]):
    print(i)
    if i <= 29:
        rnn_chla_roll_30[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
        a=torch.cat((a[:,1:],rnn_chla_roll_30[:,i].unsqueeze(dim=1)),1)
    if i > 29:
        p=torch.cat((x_test[:,0:6,i],a[:,-30:]),1)
        rnn_chla_roll_30[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
        a = torch.cat((a[:, 1:], rnn_chla_roll_30[:, i].unsqueeze(dim=1)), 1)
rnn_chla_roll_30=rnn_chla_roll_30.detach().numpy()
scio.savemat('rnn_chla_roll_30.mat', {'rnn_chla_roll_30':rnn_chla_roll_30})

#单步容忍滚动预测
a=x_test[:,:,0]
rnn_chla_roll_35=torch.Tensor(y_test.shape[0],y_test.shape[1])
for i in np.arange(y_test.shape[1]):
    print(i)
    if i <= 34:
        rnn_chla_roll_35[:,i]= torch.squeeze(model(a.unsqueeze(dim=2))[:,-1,:])
        a=torch.cat((a[:,1:],rnn_chla_roll_35[:,i].unsqueeze(dim=1)),1)
    if i > 34:
        p=torch.cat((x_test[:,0:1,i],a[:,-35:]),1)
        rnn_chla_roll_35[:, i] = torch.squeeze(model(p.unsqueeze(dim=2))[:, -1, :])
        a = torch.cat((a[:, 1:], rnn_chla_roll_35[:, i].unsqueeze(dim=1)), 1)
rnn_chla_roll_35=rnn_chla_roll_35.detach().numpy()
scio.savemat('rnn_chla_roll_35.mat', {'rnn_chla_roll_35':rnn_chla_roll_35})