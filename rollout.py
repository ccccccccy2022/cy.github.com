import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model import *
from prep_data import *
import tqdm
from model import *
import tqdm
from download_data import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn import preprocessing
""" After training the model, we can try to use the model to do
jumpy predictions.
"""
import pandas as pd
def DS(x,y):#x:p,y:t
    indexa=pd.concat([x,y],axis=1).dropna(axis=0).index
    x=x.loc[indexa]
    y=y.loc[indexa]
    assert len(x)==len(y)
    xy=pd.concat([x,y],axis=1)
    diffd = pd.concat([x,y],axis=1).diff().dropna(axis=0)
    yt_yt_1=xy.iloc[:,0]-xy.iloc[:,1].shift(1).dropna(axis=0)
    judge =diffd.iloc[:,0]*diffd.iloc[:,1]
    len(np.where(judge>0)[0])/len(judge)
    Dss =len(np.where(judge>0)[0])/len(judge)
    return Dss
def MAE(y, t):
    return 1 / len(y) * sum(abs(y - t))
def MAPE(y,t):
    if len(np.delete(np.array(y),np.where(y==0)))==len(y):
        mape_list = pd.Series(abs(y - t)/abs(y))
        med = np.median(mape_list)
        for i in range(len(mape_list)):
            if mape_list.iloc[i] > med * 5:
                mape_list.iloc[i] = med * 5
        return np.mean(mape_list)
    elif len(np.delete(np.array(y),np.where(y==0)))>0:
        y_=np.delete(np.array(y),np.where(y==0))
        t_=np.delete(np.array(t),np.where(y==0))
        mape_list = pd.Series(abs(y_ - t_)/abs(y_))
        med = np.median(mape_list)
        for i in range(len(mape_list)):
            if mape_list.iloc[i] > med * 5:
                mape_list.iloc[i] = med * 5
        return np.mean(mape_list)
    else:
        # print('MAE')
        return MAE(y,t)
#### load trained model
batch_size = 1
checkpoint = torch.load("./output/model3/new_model_epoch_3999.pt")
input_size = 400
processed_x_size = 20
belief_state_size = 20
state_size = 20
tdvae = TD_VAE(input_size,batch_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#### load dataset 
with open("./data/y.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()
tdvae = tdvae.cuda()
data = MNIST_Dataset(MNIST['test_image'], binary = False)
test_data = GetLoader(data,MNIST['test_y'].reshape(-1,1),MNIST['test_yt_1'].reshape(-1,1))

# data=MNIST['train_image']
batch_size = 1
data_loader = DataLoader(test_data ,
                         batch_size = batch_size,
                         shuffle = False)
Rollout_images=[]
fig = plt.figure(0, figsize=(12, 4))
str_code='train'
data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader),
                      bar_format="{l_bar}{r_bar}")
for i, Images in data_iter:
    images, y, yt_1 = Images[0], Images[1], Images[2]
    images = images.cuda()
    ## calculate belief
    tdvae.forward(images) 
    ## jumpy rollout
    t1, t2 = 11,12
    rollout_images,rollout_images2=tdvae.rollout(images, t1, t2,yt_1.cuda())
    Rollout_images.append(rollout_images2.detach().cpu().numpy()[0])


X_scaled =MinMaxScaler(feature_range=(min(MNIST['test_yt_1'])[0],max(MNIST['test_yt_1'])[0]))
datay = X_scaled.fit_transform(abs(np.array(Rollout_images)).reshape(-1,1))# 反归一化
plt.plot(datay)
plt.plot(MNIST['test_y'])
    ### plot results
fig = plt.figure(0, figsize = (t2+2,batch_size))
minn=min(MNIST['real_y'][:len(MNIST['train_y'])])
maxx=max(MNIST['real_y'][:len(MNIST['train_y'])])
real_price=datay*(maxx-minn)+minn
plt.plot(real_price)
plt.plot(MNIST['real_y'])#[5375:]
MAPE(real_price.flatten(),MNIST['real_y'][5375:5375+200])
