__author__ = "CuiYu"

import numpy as np
import urllib3
import gzip
import subprocess
import pickle
import pandas as pd
import datetime
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import torch
from sklearn import preprocessing
#read times series data
def percentile(data,n):
    data2=np.array(data).copy()
    P=np.arange(n)
    category =dict()
    percentage=[a*int(100/(n-1)) for a in np.arange(n-1)][1:]
    for index,p in enumerate(percentage):
        if p==percentage[0]:
            Index=list(set(np.where(data>=np.percentile(np.array(data),0))[0]).intersection( set(np.where(data<=np.percentile(np.array(data),percentage[index+1]))[0])))
            data2[Index]=index
        elif p==percentage[-1]:
            Index=list(set(np.where(data>np.percentile(np.array(data),percentage[index]))[0]).intersection( set(np.where(data<=np.percentile(np.array(data),100))[0])))
            data2[Index]=index
        else:
            Index=list(set(np.where(data>np.percentile(np.array(data),percentage[index]))[0]).intersection( set(np.where(data<=np.percentile(np.array(data),percentage[index+1]))[0])))
            data2[Index]=index
    return data2.tolist()
def compute_transition_matrix(data, n, step = 1):
    P = np.zeros((n, n))
    m = len(data)
    for i in range(m):
        initial, final = i, i + step
        if final < m:
            P[int(data[initial])][int(data[final])] += 1
    sums = np.sum(P, axis = 1)
    for i in range(n):
        if sums[i] != 0: # Added this check
            for j in range(n):
                P[i][j] = P[i][j] / sums[i]
    return P
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, data_label2):
        self.data = data_root
        self.label = data_label
        self.label2 = data_label2
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        labels2 = self.label2[index]
        return data, labels, labels2
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def data_to_label(datay1):
    dataylabel=[]
    for x in datay1:
        if np.mean(datay1)+0*np.std(datay1)<abs(x)<=np.mean(datay1)+1*np.std(datay1):
            dataylabel.append(1)
        elif np.mean(datay1)+1*np.std(datay1)<abs(x)<=np.mean(datay1)+2*np.std(datay1):
            dataylabel.append(2)
        elif np.mean(datay1)+2*np.std(datay1)<abs(x)<=np.mean(datay1)+3*np.std(datay1):
            dataylabel.append(3)
        else:
            dataylabel.append(4)
    return dataylabel
if __name__=='__main__':
    datat = pd.read_excel('D:\宏观分析\论文\Dynare-for-DSGE-Models-master\数据.xlsx', index_col=0,sheet_name='训练集')
    datatx = pd.read_excel('D:\宏观分析\论文\Dynare-for-DSGE-Models-master\数据.xlsx', index_col=0,sheet_name='验证集')
    data0=pd.concat([datat,datatx],axis=0)
    dataylabel=data_to_label(datat['y'])
    datatxlabel=data_to_label(datatx['y'])
    step2=1   #forecast_len
    perc_step=20
    data3=[]
    matrix=np.array([[]])
    for row in np.arange(0,len(data0)-step2,step2):
        data = data0.drop(columns='y').values[row:row + step2, :].flatten().tolist()
        data2 = percentile(data, perc_step)
        data3.append(percentile((data+data0['y'].values[row+step2:row + step2+step2].flatten().tolist()),perc_step)[-1])
        if row==0:
            matrix = compute_transition_matrix(data2,perc_step, step = 1).flatten()
        else:
            matrix = np.vstack([matrix,compute_transition_matrix(data2,perc_step, step = 1).flatten()])
    matrix2=matrix[:len(data0),:]
    forecast_len=1
    datay1=data0['y']

    print("Saving data into a pickle file ...")
    test_len=len(datatx)
    lena =len(datat)-test_len
    X_scaled = preprocessing.MinMaxScaler(feature_range=(0,1))
    datay=X_scaled.fit_transform(datat['y'].values.reshape(-1,1))
    X_scaled2 = preprocessing.MinMaxScaler(feature_range=(0,1))
    datayx=X_scaled2.fit_transform(datatx['y'].values.reshape(-1,1))

    data = {'train_image':  matrix2[1:lena-1,:],
            'train_label':  dataylabel,
            'test_image': matrix2[lena-1:lena-1+test_len,:],
            'test_label':datatxlabel,
            'train_y':datay,
            'train_yt_1':pd.DataFrame(datay).shift(1).fillna(0).values,
            'test_y':datayx,
            'test_yt_1':pd.DataFrame(datayx).shift(1).fillna(0).values,
            'real_y':datatx['y'],
            'scale':X_scaled2 }

    with open("./data/y.pkl", 'wb') as file_handle:
        pickle.dump(data, file_handle)

    # X_train, X_test, y_train, y_test = data['train_image'],  data['test_image'],data['train_label'],data['test_label'] # 数据集划分
    #
    # def objective(trial):
    #     param = {
    #         'metric': 'rmse',
    #         'random_state': 48,
    #         'n_estimators': 20000,
    #         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
    #         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    #         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    #         'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
    #         'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
    #         'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20, 50]),
    #         'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
    #         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    #         'cat_smooth': trial.suggest_int('cat_smooth', 1, 100)
    #     }
    #
    #     lgb = LGBMRegressor(**param)
    #     lgb.fit(X_train, y_train, verbose=False)
    #     pred_lgb = lgb.predict(X_test)
    #     rmse = mean_squared_error(y_test, pred_lgb, squared=False)
    #     return rmse
    # def objectivex(trial):
    #     param = {
    #         'learning_rate': trial.suggest_categorical('learning_rate',
    #                                                    [0.01, 0.02, 0.04, 0.16, 0.18, 0.2,0.3,0.4,0.5]),
    #         'n_estimators':trial.suggest_int('n_estimators', 100, 10000),
    #         'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
    #         'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
    #         'min_child_weight': trial.suggest_int('min_child_weight', 4,10),
    #     }
    #     model = xgb.XGBRegressor(**param)
    #     eval_set = [(X_train[30:,:], y_train[30:])]
    #     model.fit(X_train, y_train,eval_set=eval_set,eval_metric='rmse',verbose=False)
    #     pred_lgb = model.predict(X_test)
    #     rmse = mean_squared_error(y_test, pred_lgb, squared=False)
    #     if trial.should_prune():
    #         raise optuna.exceptions.TrialPruned()
    #     return rmse
    # study = optuna.create_study(direction='minimize')
    # n_trials = 50  # try50次
    # study.optimize(objectivex, n_trials=n_trials)
    # model = xgb.XGBRegressor(**study.best_params)
    # model.fit(X_train, y_train, verbose=False)
    # # save model to file
    # pickle.dump(model, open("./output/pima.pickle.dat", "wb"))
    # pred_lgb = model.predict(X_test)
    # pred_lgbt = model.predict(X_train)
    #
