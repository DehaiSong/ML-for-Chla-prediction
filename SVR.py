import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import h5py
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

#导入数据
file='svr_5paras_data.mat'
mat = h5py.File(file,'r+')
y_train=np.array(mat['y_train']).T
x_train=np.array(mat['x_train']).T
chla_test_svr=np.array(mat['chla_test_svr']).T
PN_test_svr=np.array(mat['PN_test_svr']).T
PN_test_svr2=np.array(mat['PN_test_svr2']).T

print(x_train.shape)
print(y_train.shape)

#划分数据
#x_train, x_valid, y_train, y_valid = train_test_split(PN_2017_svm, chla_2017_svm, test_size=0.3,random_state=10)
# x_train=PN_train_svr
# y_train=chla_train_svr

#数据归一化
scaler = MinMaxScaler()
x_train_scaled = scaler.fit(x_train).transform(x_train)
#x_valid_scaled = scaler.fit(x_train).transform(x_valid)
x_test_scaled = scaler.fit(x_train).transform(PN_test_svr)

#交叉验证
best_score = -10
for gamma in [0.01, 0.1, 1, 10, 100]:
    print(gamma)
    for c in [0.01, 0.1, 0.2, 1, 10]:
        print(c)
        # 对于每种参数可能的组合，进行一次训练
        svr = SVR(gamma=gamma, C=c)
        # 3 折交叉验证
        scores = cross_val_score(svr, x_train_scaled, y_train[:, 0], cv=3, scoring='neg_root_mean_squared_error')
        score = scores.mean()
        # 找到表现最好的参数
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, "C": c}

svr = SVR(**best_parameters)

model_best = svr.fit(x_train_scaled, y_train)
prediction1 = model_best.predict(x_train_scaled)
prediction2 = model_best.predict(x_test_scaled)

print(prediction1.shape)
print(prediction2.shape)

# #反归一化
# prediction1=np.expand_dims(prediction1,axis=1)
# prediction2=np.expand_dims(prediction2,axis=1)
# prediction1 = scaler.inverse_transform(prediction1)
# prediction2 = scaler.inverse_transform(prediction2)

#保存模型
joblib.dump(model_best, 'svr_5_paras.pkl')
#模型预测
x_scaled = np.zeros((PN_test_svr2.shape))
pre_Chla_svr=np.zeros((PN_test_svr2.shape[0],PN_test_svr2.shape[1]))
for i in range(PN_test_svr2.shape[1]):
    x_scaled[:, i, :] = scaler.fit(x_train).transform(PN_test_svr2[:, i, :])
    pre_Chla_svr[:,i]=model_best.predict(np.squeeze(x_scaled[:,i,:]))
scio.savemat('Chla_svr_5paras.mat', {'pre_Chla_svr':pre_Chla_svr})

print('最佳超参数:{}'.format(best_parameters))
print('交叉验证最佳RMSE:{:.2f}'.format(best_score))
#print('验证集： R2:{}\t RMSE: {}'.format(r2_score(y_valid, prediction1), np.sqrt(mean_squared_error(y_valid, prediction1))))
print('训练集： R2:{}\t RMSE: {}'.format(r2_score(y_train[:,0], prediction1),
                                     np.sqrt(mean_squared_error(y_train[:,0], prediction1))))
print('测试集： R2:{}\t RMSE: {}'.format(r2_score(chla_test_svr[:,0], prediction2),
                                     np.sqrt(mean_squared_error(chla_test_svr[:,0], prediction2))))

# svr = SVR(gamma=100, C=100)
# model_best = svr.fit(x_train_scaled, y_train[:, 0])
# prediction1 = model_best.predict(x_valid_scaled)
# prediction2 = model_best.predict(x_test_scaled)
# print('验证集： R2:{}\t RMSE: {}'.format(r2_score(y_valid, prediction1), np.sqrt(mean_squared_error(y_valid, prediction1))))
# print('测试集： R2:{}\t RMSE: {}'.format(r2_score(chla_2018_svm, prediction2),
#                                      np.sqrt(mean_squared_error(chla_2018_svm, prediction2))))



