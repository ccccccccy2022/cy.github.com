__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
import sys
import tqdm
from download_data import *
from sklearn.preprocessing import StandardScaler

#### preparing dataset
# with open("./data/MNIST.pkl", 'rb') as file_handle:
#     MNIST = pickle.load(file_handle)

with open("./data/y.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
data = MNIST_Dataset(MNIST['train_image'], binary = False)
batch_size = 128

train_data = GetLoader(data,MNIST['train_y'],MNIST['train_yt_1'])
data_loader = DataLoader(train_data ,
                         batch_size = batch_size,
                         shuffle = False)
#### build a TD-VAE model
input_size = 400
processed_x_size = 20
belief_state_size = 20
state_size = 20
tdvae = TD_VAE(input_size,batch_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda()
#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.000005)
num_epoch = 4000
log_file_handle = open("./log/loginfo_new.txt", 'w')
Loss=[]
for epoch in range(num_epoch):
    # for idx, images in enumerate(data_loader):
    str_code='train'
    data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader),
                          bar_format="{l_bar}{r_bar}")
    for i, Images in data_iter:
        images, y,yt_1 = Images[0], Images[1],Images[2]

        images = images.cuda()
        tdvae.forward(torch.tensor(images,dtype=torch.float))
        t_1 = 14-np.random.choice(10)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss = tdvae.calculate_loss(t_1, t_2)
        y=y.cuda()
        yt_1=yt_1.cuda()
        try:
            loss2, _ = tdvae.Decoder2(images, yt_1,y)
            optimizer.zero_grad()
            (loss + 50*loss2).backward()
            Loss.append((loss + loss2).detach().cpu().numpy())
            optimizer.step()
        except:
            pass
    print((loss + loss2).detach().cpu().numpy())

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "./output/model3/new_model_epoch_{}.pt".format(epoch))
