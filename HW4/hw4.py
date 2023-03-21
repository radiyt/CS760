# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:59:00 2023
@author: TY
"""
import numpy as np
import string
import math
c=list(string.ascii_lowercase)
c.append(' ')
# In[2-2 2-3]
counte=np.zeros((27,2),dtype=np.float64)
for num in range(10):
    contents=''
    with open('./languageID/e'+str(num)+'.txt', "r+") as file:
        for line in file:
            if not line.isspace():
                contents += line
    contents = contents.replace('\n','')
    for i in range(27):
        counte[i,0]+=contents.count(c[i])
counte[:,1]=(counte[:,0]+0.5)/(sum(counte[:,0])+27*0.5)

countj=np.zeros((27,2),dtype=np.float64)
for num in range(10):
    contents=''
    with open('./languageID/j'+str(num)+'.txt', "r+") as file:
        for line in file:
            if not line.isspace():
                contents += line
    contents = contents.replace('\n','')
    for i in range(27):
        countj[i,0]+=contents.count(c[i])
countj[:,1]=(countj[:,0]+0.5)/(sum(countj[:,0])+27*0.5)

counts=np.zeros((27,2),dtype=np.float64)
for num in range(10):
    contents=''
    with open('./languageID/s'+str(num)+'.txt', "r+") as file:
        for line in file:
            if not line.isspace():
                contents += line
    contents = contents.replace('\n','')
    for i in range(27):
        counts[i,0]+=contents.count(c[i])
counts[:,1]=(counts[:,0]+0.5)/(sum(counts[:,0])+27*0.5)
# In[2-4 2-5]
counte10=np.zeros((27,2),dtype=np.float64)
contents=''
with open('./languageID/e10.txt', "r+") as file:
    for line in file:
        if not line.isspace():
            contents += line
contents = contents.replace('\n','')
count=np.zeros((27,2),dtype=np.float64)

for i in range(27):
    counte10[i,0]=contents.count(c[i])
counte10[:,1]=(counte10[:,0]+0.5)/(sum(counte10[:,0])+27*0.5)

p_hat_e=np.zeros((27,1),dtype=np.float64)
p_hat_j=np.zeros((27,1),dtype=np.float64)
p_hat_s=np.zeros((27,1),dtype=np.float64)
for i in range(27):
    p_hat_e[i]=counte10[i,0]*math.log(counte[i,1])
    p_hat_j[i]=counte10[i,0]*math.log(countj[i,1])
    p_hat_s[i]=counte10[i,0]*math.log(counts[i,1])
p_hat_e_sum=sum(p_hat_e)
p_hat_j_sum=sum(p_hat_j)
p_hat_s_sum=sum(p_hat_s)

# In[2-6 2-5]
from math import e
p_e=p_hat_e_sum*math.log(1/3)/(len(contents)*math.log(1/27))
p_e2=p_hat_e_sum+math.log(1/3)-len(contents)*math.log(1/27)

p_e=e**(p_e)
p_j=p_hat_j_sum*math.log(1/3)/(len(contents)*math.log(1/27))
p_j=e**(p_j)
p_s=p_hat_s_sum*math.log(1/3)/(len(contents)*math.log(1/27))
p_s=e**(p_s)
# In[2-7]
predictions=[]
for lan in ['e','j','s']:
    for num in range(10, 20):
        countval=np.zeros((27,2),dtype=np.float64)
        contents=''
        with open('./languageID/'+lan+str(num)+'.txt', "r+") as file:
            for line in file:
                if not line.isspace():
                    contents += line
        contents = contents.replace('\n','')
        for i in range(27):
            countval[i,0]+=contents.count(c[i])
        countval[:,1]=(countval[:,0]+0.5)/(sum(countval[:,0])+27*0.5)
        
        p_hat_e=np.zeros((27,1),dtype=np.float64)
        p_hat_j=np.zeros((27,1),dtype=np.float64)
        p_hat_s=np.zeros((27,1),dtype=np.float64)
        for i in range(27):
            p_hat_e[i]=countval[i,0]*math.log(counte[i,1])
            p_hat_j[i]=countval[i,0]*math.log(countj[i,1])
            p_hat_s[i]=countval[i,0]*math.log(counts[i,1])
        p_hat_e_sum=sum(p_hat_e)
        p_hat_j_sum=sum(p_hat_j)
        p_hat_s_sum=sum(p_hat_s)
        m=max(p_hat_e_sum,p_hat_j_sum,p_hat_s_sum)
        if p_hat_e_sum==m:
            predictions.append('e')
        elif p_hat_j_sum==m:
            predictions.append('j')
        else:
            predictions.append('s')
# In[2-8]
predictions=[]
for lan in ['e','j','s']:
    for num in range(10, 20):
        countval=np.zeros((27,2),dtype=np.float64)
        contents=''
        with open('./languageID/'+lan+str(num)+'.txt', "r+") as file:
            for line in file:
                if not line.isspace():
                    contents += line
        contents = contents.replace('\n','')
        for i in range(27):
            countval[i,0]+=contents.count(c[i])
        countval[:,1]=(countval[:,0]+0.5)/(sum(countval[:,0])+27*0.5)
        
        p_hat_e=np.zeros((27,1),dtype=np.float64)
        p_hat_j=np.zeros((27,1),dtype=np.float64)
        p_hat_s=np.zeros((27,1),dtype=np.float64)
        for i in range(27):
            p_hat_e[i]=countval[i,0]*math.log(counte[i,1])
            p_hat_j[i]=countval[i,0]*math.log(countj[i,1])
            p_hat_s[i]=countval[i,0]*math.log(counts[i,1])
        p_hat_e_sum=sum(p_hat_e)
        p_hat_j_sum=sum(p_hat_j)
        p_hat_s_sum=sum(p_hat_s)
        m=max(p_hat_e_sum,p_hat_j_sum,p_hat_s_sum)
        if p_hat_e_sum==m:
            predictions.append('e')
        elif p_hat_j_sum==m:
            predictions.append('j')
        else:
            predictions.append('s')
# In[3-2]
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, X, y, batch = 128, lr = 5e-3,  epochs = 30):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        
        self.x = []
        self.y = []
        self.loss = []
        self.acc = []

        self.init_weights()
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    def d_sigmoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    def softmax(self, z):
      z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
      return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
  
    def init_weights(self):
        self.W1 = np.random.randn(self.input.shape[1],300)
        self.W2 = np.random.randn(self.W1.shape[1],200)
        self.W3 = np.random.randn(self.W2.shape[1],10)

    def feedforward(self):
        assert self.x.shape[1] == self.W1.shape[0]
        self.z1 = self.x.dot(self.W1)
        self.a1 = self.sigmoid(self.z1)
    
        assert self.a1.shape[1] == self.W2.shape[0]
        self.z2 = self.a1.dot(self.W2)
        self.a2 = self.sigmoid(self.z2)
    
        assert self.a2.shape[1] == self.W3.shape[0]
        self.z3 = self.a2.dot(self.W3)
        self.a3 = self.softmax(self.z3)
        self.error = self.a3 - self.y
    def backprop(self):
        dcost = (1/self.batch)*self.error
    
        DW3 = np.dot(dcost.T,self.a2).T
        DW2 = np.dot((np.dot((dcost),self.W3.T) * self.d_sigmoid(self.z2)).T,self.a1).T
        DW1 = np.dot((np.dot(np.dot((dcost),self.W3.T)*self.d_sigmoid(self.z2),self.W2.T)*self.d_sigmoid(self.z1)).T,self.x).T

        assert DW3.shape == self.W3.shape
        assert DW2.shape == self.W2.shape
        assert DW1.shape == self.W1.shape

        self.W3 = self.W3 - self.lr * DW3
        self.W2 = self.W2 - self.lr * DW2
        self.W1 = self.W1 - self.lr * DW1

    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
        
    def train(self):
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()
    
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end]
                self.y = self.target[start:end]
                self.feedforward()
                self.backprop()
                l+=np.mean(self.error**2)
                acc+= np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.batch
            losstemp=l/(self.input.shape[0]//self.batch)
            self.loss.append(losstemp)
            self.acc.append(acc*100/(self.input.shape[0]//self.batch))
            print ("epoch:  % i, loss: % f" % (epoch, losstemp))
            np.save('./weights/W1_% i.npy'%epoch,self.W1)
            np.save('./weights/W2_% i.npy'%epoch,self.W2)
            np.save('./weights/W3_% i.npy'%epoch,self.W3)
            
    def plot(self):
        plt.figure(dpi = 200)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        
    def test(self,xtest,ytest):
        self.x = xtest
        self.y = ytest
        self.feedforward()
        acc = np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
        print("Error:", 100 * (1-acc), "%")
def one_hot(y):
    table = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        table[i][int(y[i][0])] = 1 
    return table

train_data = MNIST(".", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
x_train=train_data.data
x_train = torch.flatten(x_train,1)
x_train=np.array(x_train/255)

y_train=train_data.targets
y_train=np.expand_dims(y_train, axis=1)
y_train=one_hot(y_train)

x_test=test_data.data
x_test = torch.flatten(x_test,1)
x_test=np.array(x_test/255)

y_test=test_data.targets
y_test=np.expand_dims(y_test, axis=1)
y_test=one_hot(y_test)
        
NN = NeuralNetwork(x_train, y_train, batch = 128, lr = 5e-3, epochs = 30) 
NN.train()
NN.plot()
NN.test(x_test,y_test)

# In[3-3 3-4 train]
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import statistics
import matplotlib.pyplot as plt

class NN3layer(nn.Module):
    def __init__(self):
        super(NN3layer, self).__init__()
        self.fc1 = nn.Linear(28*28, 300,bias=False)
        self.fc2 = nn.Linear(300, 200,bias=False)
        self.fc3 = nn.Linear(200, 10,bias=False)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight,0) 
                # m.weight.data=torch.rand(m.weight.data.shape)*2-1
                nn.init.normal_(m.weight) 

BS=256
MAX_EPOCH=50

net=NN3layer()
net.initialize()
train_data = MNIST(".", train=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BS, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-1)

# train
train_loss=[]
for epoch in range(MAX_EPOCH):
    # train_loss = 0.0
    sumloss=0
    train_correct=0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = torch.flatten(inputs,1)
        # labels = F.one_hot(labels)
        # zeros the paramster gradients
        optimizer.zero_grad()#
        net.train()
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        sumloss+=loss
        # print statistics
        train_loss.append(str(loss.item()))
        if i!=0 and i % 200 == 0:
            print("Training2:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch, MAX_EPOCH, i + 1, len(train_loader), sumloss/(i*BS)))
    torch.save(net.state_dict(), './weights/weights_{:0>3}_{:.4f}.pth'.format(epoch,sumloss/60000))

# In[3-3 3-4 test]
test_data = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BS, shuffle=False)   

modelpath='./weights/weights_019_0.0013.pth'
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