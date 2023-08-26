import torch
import h5py
import pandas as pd
import numpy as np
import joblib
import scipy.io as scio
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sklearn
import joblib
import shap
import matplotlib.pyplot as plt


# 导入数据
file='forest_data_5paras_train.mat'
mat = h5py.File(file,'r+')
chla_train_forest=np.array(mat['chla_train_forest']).T
PN_train_forest=np.array(mat['PN_train_forest']).T
chla_test_forest=np.array(mat['chla_test_forest']).T
PN_test_forest=np.array(mat['PN_test_forest']).T
PN_test_forest2=np.array(mat['PN_test_forest2']).T
#
# # chla_2017_forest=np.log10(chla_2017_forest)
# # chla_2017_forest[np.isinf(chla_2017_forest)]=0
# # chla_2018_forest=np.log10(chla_2018_forest)
# # chla_2018_forest[np.isinf(chla_2018_forest)]=0
#
# print(chla_test_forest.shape)
# print(PN_test_forest.shape)
x_train, x_valid, y_train, y_valid = train_test_split(PN_train_forest, chla_train_forest, test_size=0.3,random_state=20)
x_train=PN_train_forest
y_train=chla_train_forest

# # 模型训练
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
parameter={'n_estimators':np.linspace(20,500,10,dtype=int),'max_depth':np.linspace(10,50,5,dtype=int),'random_state':[42]}
model = GridSearchCV(RandomForestRegressor(),parameter,scoring='r2',
                       cv=3, verbose=10,n_jobs=-1)
model.fit(x_train, y_train)

model_best = model.best_estimator_
print('最佳参数组合: ',model.best_params_)
print('训练集交叉验证的最佳R2: ',abs(model.best_score_))
evaluate(model_best, x_train, y_train)
evaluate(model_best, x_valid, y_valid)
# evaluate(model_best, PN_2018_forest, chla_2018_forest)

prediction3 = model_best.predict(x_train)
prediction1 = model_best.predict(x_valid)
prediction2 = model_best.predict(PN_test_forest)
print('训练集： R2:{}\t RMSE: {}'.format(r2_score(y_train, prediction3), np.sqrt(mean_squared_error(y_train, prediction3))))
print('测试集： R2:{}\t RMSE: {}'.format(r2_score(chla_test_forest, prediction2),
                                     np.sqrt(mean_squared_error(chla_test_forest, prediction2))))

#保存模型
#joblib.dump(model_best, 'forest_daliy_5paras(2).pkl')
model_best=joblib.load( 'forest_daliy_5paras.pkl')

pre_Chla_forest=np.zeros((chla_train_forest.shape[0],chla_train_forest.shape[1]))
for i in range(chla_train_forest.shape[1]):
    pre_Chla_forest[:,i]=model_best.predict(np.squeeze(PN_train_forest[:,i,:]))
scio.savemat('Chla_forest_5paras_train.mat', {'pre_Chla_forest':pre_Chla_forest})




# 解释器
# file='forest_data_5paras.mat'
# mat = h5py.File(file,'r+')
# x_test=np.array(mat['PN_test_forest']).T


# mat2=scio.loadmat('/lustre/home/songdehai/lihai/matlab/forest_red_tide.mat')
# x_test=np.array(mat2['forest_red_tide'])
#
# model=joblib.load( 'forest_daliy_5paras.pkl')
# explainer = shap.TreeExplainer(model)
# x_test = pd.DataFrame(x_test) #将numpy的array数组x_test转为dataframe格式。
# x_test.columns = ['T','S','DIN','DOP','ZPT'] #添加特征名称
# shap_values = explainer.shap_values(x_test) #x_test为特征参数数组 shap_value为解释器计算的shap值
# f = plt.figure()
# shap.summary_plot(shap_values, x_test)
# f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)
# b = plt.figure()
# shap.summary_plot(shap_values, x_test, plot_type="bar")
# b.savefig("summary_plot2.png", bbox_inches='tight', dpi=600)