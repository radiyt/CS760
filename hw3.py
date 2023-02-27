# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:49:37 2023
@author: TongYu
"""
import numpy as np
import matplotlib.pyplot as plt
import math
# In[1.2.(a)]
a=np.array([[0,3,0],[2,0,0],[0,1,3],[0,1,2],[-1,0,1],[1,1,1]])
x=np.array([0,0,0])
edis=np.zeros(6)
for i in range(6):
    edis[i]=math.sqrt(sum((x-a[i,:])**2))

# In[1.5.(a)]
xplot=np.array([0,0.25,0.5,1])
yplot=np.array([0.33333,0.66667,1,1])
plt.xlabel("False positive rate")
plt.ylabel("True positiave rate")
plt.scatter(xplot,yplot , c='r', alpha=0.5)
plt.plot(xplot, yplot, color='green', linestyle='-', linewidth=2)

# In[2.1 programming]
def knn(x1,y,x2,k):
    r,c=x1.shape
    r2,c2=x2.shape
    if c!=c2:
        print('error')
        return
    dis=np.zeros((r,r2))
    ycandidate=np.zeros((r2,k),dtype=np.int8)
    output=np.zeros((r2,1))
    for i in range(r2):
        distemp=np.sum((x1-x2[i,:])**2,axis=1)
        dis[:,i]=distemp
        idx = np.argpartition(distemp, k)
        ytemp=np.array(y[idx[:k]],dtype=np.int8)
        ycandidate[i,:]=ytemp
        output[i]=np.bincount(ytemp[0]).argmax()
    return output   

with open('./data/D2z.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])

# draw grid
xlist = np.arange(-2, 2, 0.1)
ylist = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(xlist, ylist)
m=X.flatten()
num=m.size
m2=Y.flatten()
mm=np.vstack((m,m2))
ypre=knn(x,y,mm.transpose(),1)
ypre=ypre.reshape(X.shape)
index1=np.where(ypre==0)
index2=np.where(ypre==1)
plt.scatter(X[index1],Y[index1],c='b',marker="+",linewidths=0.5)
plt.scatter(X[index2],Y[index2],c='r',marker="+",linewidths=0.5) 
# plt.show()
# draw train set    
index1=np.where(y==0)[0]
index2=np.where(y==1)
plt.scatter(x[index1,0],x[index1,1],marker="^",facecolors='none', edgecolors='k')
plt.scatter(x[index2,0],x[index2,1],marker="o",facecolors='none', edgecolors='k')
plt.show()

# In[2. Spam filter]
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def knn(x1,y,x2,k):
    r,c=x1.shape
    r2,c2=x2.shape
    if c!=c2:
        print('error')
        return
    dis=np.zeros((r,r2))
    ycandidate=np.zeros((r2,k),dtype=np.int8)
    output=np.zeros((r2,1))
    for i in tqdm(range(r2)):
        distemp=np.sum((x1-x2[i,:])**2,axis=1)
        dis[:,i]=distemp
        idx = np.argpartition(distemp, k)
        ytemp=np.array(y[idx[:k]],dtype=np.int8)
        ycandidate[i,:]=ytemp
        output[i]=np.bincount(ytemp).argmax()
    return output   

import pandas as pd
data = pd.read_csv("./data/emails.csv", delimiter= ',').to_numpy()

feature=np.array(data[:,1:3001], dtype=np.int64)
predict=np.array(data[:,3001], dtype=np.int8)

feature1=feature[0:1000,:]
feature2=feature[1000:2000,:]
feature3=feature[2000:3000,:]
feature4=feature[3000:4000,:]
feature5=feature[4000:5000,:]

predict1=predict[0:1000]
predict2=predict[1000:2000]
predict3=predict[2000:3000]
predict4=predict[3000:4000]
predict5=predict[4000:5000]

trainx1=np.vstack((feature2,feature3,feature4,feature5))
trainy1=np.hstack((predict2,predict3,predict4,predict5))
testx1=feature1
testy1=predict1

trainx2=np.vstack((feature1,feature3,feature4,feature5))
trainy2=np.hstack((predict1,predict3,predict4,predict5))
testx2=feature2
testy2=predict2

trainx3=np.vstack((feature1,feature2,feature4,feature5))
trainy3=np.hstack((predict1,predict2,predict4,predict5))
testx3=feature3
testy3=predict3

trainx4=np.vstack((feature1,feature2,feature3,feature5))
trainy4=np.hstack((predict1,predict2,predict3,predict5))
testx4=feature4
testy4=predict4

trainx5=np.vstack((feature1,feature2,feature3,feature4))
trainy5=np.hstack((predict1,predict2,predict3,predict4))
testx5=feature5
testy5=predict5
# In[2.2]
#run each fold one by one  
# fold 1
y1=knn(trainx1,trainy1,testx1,1)
tn, fp, fn, tp = confusion_matrix(testy1, y1).ravel()

# fold 2
y2=knn(trainx2,trainy2,testx2,1)
tn, fp, fn, tp = confusion_matrix(testy2, y2).ravel()

# fold 3
y3=knn(trainx3,trainy3,testx3,1)
tn, fp, fn, tp = confusion_matrix(testy3, y3).ravel()

# fold 4
y4=knn(trainx4,trainy4,testx4,1)
tn, fp, fn, tp = confusion_matrix(testy4, y4).ravel()

# fold 5
y5=knn(trainx5,trainy5,testx5,1)
tn, fp, fn, tp = confusion_matrix(testy5, y5).ravel()

accuracy=(tn+tp)/(tn+fp+fn+tp)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
print(accuracy,precision,recall)
# In[2.3]
from scipy.special import expit
def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def calgradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    db = (1/m)*np.sum((y_hat - y)) 
    return dw, db
def logiregre_train(X, y, bs, epochs, lr):
    m, n = X.shape
    w = np.zeros((n,1))
    b = 0
    y = y.reshape(m,1)

    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            y_hat = expit(np.dot(xb, w) + b)
            dw, db = calgradients(xb, yb, y_hat)

            w = w-lr*dw
            b = b-lr*db
        print(loss(y, expit(np.dot(X, w) + b)))
    return w, b

def predict(X,w,b):
    preds = expit(np.dot(X, w) + b)
    y = []
    y = [1 if i > 0.5 else 0 for i in preds]
    return np.array(y)
#run each fold one by one    
# fold 1
w, b = logiregre_train(trainx1, trainy1, bs=100, epochs=100, lr=0.0001)
y1=predict(testx1,w,b)
tn, fp, fn, tp = confusion_matrix(testy1, y1).ravel()
# fold 2
w, b = logiregre_train(trainx2, trainy2, bs=100, epochs=100, lr=0.0001)
y2=predict(testx2,w,b)
tn, fp, fn, tp = confusion_matrix(testy2, y2).ravel()
# fold 3
w, b = logiregre_train(trainx3, trainy3, bs=100, epochs=200, lr=0.0001)
y3=predict(testx3,w,b)
tn, fp, fn, tp = confusion_matrix(testy3, y3).ravel()
#fold 4
w, b = logiregre_train(trainx4, trainy4, bs=100, epochs=100, lr=0.0001)
y4=predict(testx4,w,b)
tn, fp, fn, tp = confusion_matrix(testy4, y4).ravel()
# fold 5
w, b = logiregre_train(trainx5, trainy5, bs=100, epochs=100, lr=0.0001)
y5=predict(testx5,w,b)
tn, fp, fn, tp = confusion_matrix(testy5, y5).ravel()

accuracy=(tn+tp)/(tn+fp+fn+tp)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
print(accuracy,precision,recall)
# In[2.4]
acc=np.zeros((5,2))
acc[:,0]=[1,3,5,7,10]
i=0
for k in [1,3,5,7,10]:
    y1=knn(trainx1,trainy1,testx1,k)
    tn, fp, fn, tp = confusion_matrix(testy1, y1).ravel()
    accuracy1=(tn+tp)/(tn+fp+fn+tp)
    
    y2=knn(trainx2,trainy2,testx2,k)
    tn2, fp2, fn2, tp2 = confusion_matrix(testy2, y2).ravel()
    accuracy2=(tn2+tp2)/(tn2+fp2+fn2+tp2)
    
    y3=knn(trainx3,trainy3,testx3,k)
    tn3, fp3, fn3, tp3 = confusion_matrix(testy3, y3).ravel()
    accuracy3=(tn3+tp3)/(tn3+fp3+fn3+tp3)
    
    y4=knn(trainx4,trainy4,testx4,k)
    tn4, fp4, fn4, tp4 = confusion_matrix(testy4, y4).ravel()
    accuracy4=(tn4+tp4)/(tn4+fp4+fn4+tp4)
    
    y5=knn(trainx5,trainy5,testx5,k)
    tn5, fp5, fn5, tp5 = confusion_matrix(testy5, y5).ravel()
    accuracy5=(tn5+tp5)/(tn5+fp5+fn5+tp5)
    
    acc[i,1]=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5
    print(acc[i,1])
    i=i+1
plt.scatter(acc[:,0],acc[:,1],c='b')
plt.plot(acc[:,0], acc[:,1], color='green', linestyle='-', linewidth=2)
plt.xlabel("k")
plt.ylabel("average accuracy")
plt.title("knn 5-fold corss validation")
plt.show()
# In[2.5]
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

data = pd.read_csv("./data/emails.csv", delimiter= ',').to_numpy()
feature=np.array(data[:,1:3001], dtype=np.int64)
predict=np.array(data[:,3001], dtype=np.int8)
trainx=feature[0:4000,:]
trainy=predict[0:4000]
testx=feature[4000:5000,:]
testy=predict[4000:5000]

def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss
def calgradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    db = (1/m)*np.sum((y_hat - y)) 
    return dw, db
def logiregre_train(X, y, bs, epochs, lr):
    m, n = X.shape
    w = np.zeros((n,1))
    b = 0
    y = y.reshape(m,1)
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            y_hat = expit(np.dot(xb, w) + b)
            dw, db = calgradients(xb, yb, y_hat)

            w = w-lr*dw
            b = b-lr*db
        print(loss(y, expit(np.dot(X, w) + b)))
    return w, b
def predict(X,w,b):
    preds = expit(np.dot(X, w) + b)
    y = []
    y = [1 if i > 0.5 else 0 for i in preds]
    return np.array(y),preds
def knn(x1,y,x2,k):
    r,c=x1.shape
    r2,c2=x2.shape
    if c!=c2:
        print('error')
        return
    dis=np.zeros((r,r2))
    ycandidate=np.zeros((r2,k),dtype=np.int8)
    output=np.zeros((r2,1))
    for i in tqdm(range(r2)):
        distemp=np.sum((x1-x2[i,:])**2,axis=1)
        dis[:,i]=distemp
        idx = np.argpartition(distemp, k)
        ytemp=np.array(y[idx[:k]],dtype=np.int8)
        ycandidate[i,:]=ytemp
        output[i]=np.bincount(ytemp).argmax()
    return output
def drawROC(y,ypre,conf,withconf=True):
    t_all=len(np.where(y==1)[0])
    f_all=len(np.where(y==0)[0])
    tcount=0
    fcount=0
    listxy=[]
    if withconf==True:
        order = np.argsort(conf[:,0])[::-1]
        ysort = y[order]
        for i in range(ysort.size):
            if ysort[i]==1:
                tcount+=1
                if ysort[i+1]!=1:
                    listxy.append([fcount/f_all,tcount/t_all])
            elif ysort[i]==0:
                fcount+=1
    elif withconf==False:
        order = np.argsort(ypre[:,0])[::-1]
        ysort = y[order]
        for i in range(ysort.size):
            if ysort[i]==1:
                tcount+=1
                if ysort[i+1]!=1:
                    listxy.append([fcount/f_all,tcount/t_all])
            elif ysort[i]==0:
                fcount+=1
    return listxy

w, b = logiregre_train(trainx, trainy, bs=100, epochs=100, lr=0.0001)
ylog,conf=predict(testx,w,b)

pointslog=np.array(drawROC(testy,ylog,conf,True))
plt.plot(pointslog[:,0], pointslog[:,1], color='red', linestyle='-', linewidth=2,label='Logistic')

yknn=knn(trainx, trainy,testx,5)
pointsknn=drawROC(testy,yknn,conf,False)
pointsknn=np.array(drawROC(testy,yknn,[],False))

plt.plot(pointsknn[:,0], pointsknn[:,1], color='blue', linestyle='-', linewidth=2,label='KNN')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc='lower right')
plt.show()
