# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:59:00 2023
@author: TY
"""
import numpy as np
import string
import math
#
# c=list(string.ascii_lowercase)
# c.append(' ')
# In[2-2 2-3]
# counte=np.zeros((27,2),dtype=np.float64)
# for num in range(10):
#     contents=''
#     with open('./languageID/e'+str(num)+'.txt', "r+") as file:
#         for line in file:
#             if not line.isspace():
#                 contents += line
#     contents = contents.replace('\n','')
#     for i in range(27):
#         counte[i,0]+=contents.count(c[i])
# counte[:,1]=(counte[:,0]+0.5)/(sum(counte[:,0])+27*0.5)

# countj=np.zeros((27,2),dtype=np.float64)
# for num in range(10):
#     contents=''
#     with open('./languageID/j'+str(num)+'.txt', "r+") as file:
#         for line in file:
#             if not line.isspace():
#                 contents += line
#     contents = contents.replace('\n','')
#     for i in range(27):
#         countj[i,0]+=contents.count(c[i])
# countj[:,1]=(countj[:,0]+0.5)/(sum(countj[:,0])+27*0.5)

# counts=np.zeros((27,2),dtype=np.float64)
# for num in range(10):
#     contents=''
#     with open('./languageID/s'+str(num)+'.txt', "r+") as file:
#         for line in file:
#             if not line.isspace():
#                 contents += line
#     contents = contents.replace('\n','')
#     for i in range(27):
#         counts[i,0]+=contents.count(c[i])
# counts[:,1]=(counts[:,0]+0.5)/(sum(counts[:,0])+27*0.5)
# In[2-4 2-5]
# counte10=np.zeros((27,2),dtype=np.float64)
# contents=''
# with open('./languageID/e10.txt', "r+") as file:
#     for line in file:
#         if not line.isspace():
#             contents += line
# contents = contents.replace('\n','')
# count=np.zeros((27,2),dtype=np.float64)

# for i in range(27):
#     counte10[i,0]=contents.count(c[i])
# counte10[:,1]=(counte10[:,0]+0.5)/(sum(counte10[:,0])+27*0.5)

# p_hat_e=np.zeros((27,1),dtype=np.float64)
# p_hat_j=np.zeros((27,1),dtype=np.float64)
# p_hat_s=np.zeros((27,1),dtype=np.float64)
# for i in range(27):
#     p_hat_e[i]=counte10[i,0]*math.log(counte[i,1])
#     p_hat_j[i]=counte10[i,0]*math.log(countj[i,1])
#     p_hat_s[i]=counte10[i,0]*math.log(counts[i,1])
# p_hat_e_sum=sum(p_hat_e)
# p_hat_j_sum=sum(p_hat_j)
# p_hat_s_sum=sum(p_hat_s)

# # In[2-6 2-5]
# from math import e
# p_e=p_hat_e_sum*math.log(1/3)/(len(contents)*math.log(1/27))
# p_e2=p_hat_e_sum+math.log(1/3)-len(contents)*math.log(1/27)

# p_e=e**(p_e)
# p_j=p_hat_j_sum*math.log(1/3)/(len(contents)*math.log(1/27))
# p_j=e**(p_j)
# p_s=p_hat_s_sum*math.log(1/3)/(len(contents)*math.log(1/27))
# p_s=e**(p_s)
# In[2-7]
# predictions=[]
# for lan in ['e','j','s']:
#     for num in range(10, 20):
#         countval=np.zeros((27,2),dtype=np.float64)
#         contents=''
#         with open('./languageID/'+lan+str(num)+'.txt', "r+") as file:
#             for line in file:
#                 if not line.isspace():
#                     contents += line
#         contents = contents.replace('\n','')
#         for i in range(27):
#             countval[i,0]+=contents.count(c[i])
#         countval[:,1]=(countval[:,0]+0.5)/(sum(countval[:,0])+27*0.5)
        
#         p_hat_e=np.zeros((27,1),dtype=np.float64)
#         p_hat_j=np.zeros((27,1),dtype=np.float64)
#         p_hat_s=np.zeros((27,1),dtype=np.float64)
#         for i in range(27):
#             p_hat_e[i]=countval[i,0]*math.log(counte[i,1])
#             p_hat_j[i]=countval[i,0]*math.log(countj[i,1])
#             p_hat_s[i]=countval[i,0]*math.log(counts[i,1])
#         p_hat_e_sum=sum(p_hat_e)
#         p_hat_j_sum=sum(p_hat_j)
#         p_hat_s_sum=sum(p_hat_s)
#         m=max(p_hat_e_sum,p_hat_j_sum,p_hat_s_sum)
#         if p_hat_e_sum==m:
#             predictions.append('e')
#         elif p_hat_j_sum==m:
#             predictions.append('j')
#         else:
#             predictions.append('s')
# In[2-8]
# predictions=[]
# for lan in ['e','j','s']:
#     for num in range(10, 20):
#         countval=np.zeros((27,2),dtype=np.float64)
#         contents=''
#         with open('./languageID/'+lan+str(num)+'.txt', "r+") as file:
#             for line in file:
#                 if not line.isspace():
#                     contents += line
#         contents = contents.replace('\n','')
#         for i in range(27):
#             countval[i,0]+=contents.count(c[i])
#         countval[:,1]=(countval[:,0]+0.5)/(sum(countval[:,0])+27*0.5)
        
#         p_hat_e=np.zeros((27,1),dtype=np.float64)
#         p_hat_j=np.zeros((27,1),dtype=np.float64)
#         p_hat_s=np.zeros((27,1),dtype=np.float64)
#         for i in range(27):
#             p_hat_e[i]=countval[i,0]*math.log(counte[i,1])
#             p_hat_j[i]=countval[i,0]*math.log(countj[i,1])
#             p_hat_s[i]=countval[i,0]*math.log(counts[i,1])
#         p_hat_e_sum=sum(p_hat_e)
#         p_hat_j_sum=sum(p_hat_j)
#         p_hat_s_sum=sum(p_hat_s)
#         m=max(p_hat_e_sum,p_hat_j_sum,p_hat_s_sum)
#         if p_hat_e_sum==m:
#             predictions.append('e')
#         elif p_hat_j_sum==m:
#             predictions.append('j')
#         else:
#             predictions.append('s')
# In[3-2]
# import numpy as np
# import torch
# import torchvision
# from torchvision.datasets import MNIST
# from torch.utils import data
# # Activation Functions
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))
# def d_sigmoid(x):
#     return (1 - sigmoid(x)) * sigmoid(x)

# def softmax(x):
#     exps = np.exp(x)
#     return exps / np.sum(exps)
# # def d_softmax(x): # Best implementation (VERY FAST)
# #     '''Returns the jacobian of the Softmax function for the given set of inputs.
# #     Inputs:
# #     x: should be a 2d array where the rows correspond to the samples
# #         and the columns correspond to the nodes.
# #     Returns: jacobian
# #     '''
# #     s = softmax(x)
# #     s = s.reshape((1, 10))
# #     a = np.eye(s.shape[-1])
# #     temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
# #     temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
# #     # Einsum is unsupported with Numba (nopython mode)
# #     # temp1 = np.einsum('ij,jk->ijk',s,a)
# #     # temp2 = np.einsum('ij,ik->ijk',s,s)
# #     for i in range(s.shape[0]):
# #         for j in range(s.shape[1]):
# #             for k in range(s.shape[1]):
# #                 temp1[i,j,k] = s[i,j]*a[j,k]
# #                 temp2[i,j,k] = s[i,j]*s[i,k]
     
# #     return temp1-temp2
# def d_softmax(x,dy):
#     y=torch.softmax(x, 1)
#     d1 = torch.einsum('ca,cb->cab', [y, y])
#     d2 = torch.diag_embed(y, dim1=-2, dim2=-1)
#     d = d2-d1
#     ret = torch.einsum('ab,abc->ac', dy, d)
#     return ret

# # Loss Functions 
# def celoss(y, a):
#     return -y*np.log(a)
# def d_celoss(y, a):
#     return -y/a

# # The layer class
# class Layer:
#     activationFunctions = {
#         'sigmoid': (sigmoid, d_sigmoid),
#         'softmax': (softmax, d_softmax)
#     }
#     learning_rate = 0.001

#     def __init__(self, inputs, neurons, activation):
#         self.W = np.random.rand(neurons, inputs)
#         # self.b = np.zeros((neurons, 1))
#         self.act, self.d_act = self.activationFunctions.get(activation)

#     def feedforward(self, A_prev):
#         self.A_prev = A_prev
#         self.Z = np.dot(self.W, self.A_prev)# + self.b
#         # if self.act==softmax:
#         #     self.Z=torch.from_numpy(self.Z)
#         self.A = self.act(self.Z)
#         return self.A

#     def backprop(self, dA):
#         dZ = np.multiply(self.d_act(self.Z), dA)
#         dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
#         # db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
#         dA_prev = np.dot(self.W.T, dZ)

#         self.W = self.W - self.learning_rate * dW
#         # self.b = self.b - self.learning_rate * db
#         return dA_prev

# # train
# BS=1
# train_data = MNIST(".", train=True, download=True, transform=torchvision.transforms.ToTensor())
# test_data = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
# # x_train=train_data.data
# # y_train=train_data.targets
# # x_test=test_data.data
# # y_test=test_data.targets
# train_loader = data.DataLoader(train_data, batch_size=BS, shuffle=True)
# test_loader = data.DataLoader(test_data, batch_size=BS, shuffle=True)   

# # m = 4
# epochs = 10
# layers = [Layer(28*28, 300, 'sigmoid'), Layer(300, 200, 'sigmoid'), Layer(200, 10, 'softmax')]
# lossplot = [] # to plot graph 
# loss=0
# dloss=0
# for epoch in range(epochs):
#     # Feedforward
#     for id, data in enumerate(train_loader):
#         train_features, train_labels = data
#         train_features=train_features.flatten()
#         train_labels = torch.nn.functional.one_hot(train_labels, 10).type(torch.FloatTensor)
#         train_labels=np.array(train_labels)
#         for layer in layers:
#             train_features = layer.feedforward(train_features)
#         loss =np.sum(celoss(train_labels, train_features))
#         dloss = d_celoss(train_labels, train_features)
#         # if id!=0 and id%8==0:
#         lossplot.append(loss)
#         # Backpropagation
#         for layer in reversed(layers):
#             dloss = layer.backprop(dloss)
#            # dloss=0
#            # loss=0
# # # Making predictions
# # A = x_train
# # for layer in layers:
# #     A = layer.feedforward(A)
# # print(A)
 
# # net = NN(28*28, 300, 200, 10)
# #     for _ in range(10):
# #         for i, (x, y) in enumerate(ds):
# #             yhat = net.forward(x)
# #             label = torch.nn.functional.one_hot(y, 10).type(torch.FloatTensor)
# #             loss, dy = cross_entropy(label, yhat)
# #             if i % 20 == 0:
# #                 loss = float(loss)
# #                 lss.append(loss)
# #                 pred = torch.argmax(yhat, 1)
# #                 accuracy = float(torch.sum(y == pred)) / BATCH_SIZE
# #                 acc.append(accuracy)
# #                 print(f"{loss=}, {accuracy=}")
# #                 # print(net.layers[1].y)
# #             net.backward(dy)
# In[3-3]
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import statistics

class NN3layer(nn.Module):
    def __init__(self):
        super(NN3layer, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight,0) 
                nn.init.constant_(m.weight,2*np.random.random()-1)

BS=128
MAX_EPOCH=20

net=NN3layer()
# net.initialize()
# # weight_init(net)
# train_data = MNIST(".", train=True, transform=torchvision.transforms.ToTensor())
# train_loader = data.DataLoader(train_data, batch_size=BS, shuffle=True)

# criterion = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(net.parameters(), lr=1e-03)
# optimizer = optim.AdamW(net.parameters(), lr=1e-03, weight_decay=1e-05)

# # train
# train_loss=[]
# for epoch in range(MAX_EPOCH):
#     # train_loss = 0.0
#     train_correct=0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = torch.flatten(inputs,1)
#         # labels = F.one_hot(labels)
#         # zeros the paramster gradients
#         optimizer.zero_grad()#
#         net.train()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels.long())
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         train_loss.append(str(loss.item()))
#         if i % 600 == 0:
#             print("Training2:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
#                 epoch, MAX_EPOCH, i + 1, len(train_loader), loss))
#     torch.save(net.state_dict(), './weights/weights_{:0>3}_{:.4f}.pth'.format(epoch,statistics.mean(np.array(train_loss,dtype=np.float32))))

# test
test_data = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = data.DataLoader(test_data, batch_size=BS, shuffle=False)   

modelpath='./weights/weights_019_2.3160.pth'
params = torch.load(modelpath)
net.load_state_dict(params)
del params

net.eval()
correct_num = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data 
        inputs = torch.flatten(inputs,1)
        
        outputs= net(inputs) 

        outputs = outputs.numpy()
        labels = labels.numpy()

        outputs = np.argmax(outputs, axis=1)
        correct_num += int(sum(labels == outputs))
                
    accuracy = correct_num / 10000
    print('testerror='+str(1-accuracy))