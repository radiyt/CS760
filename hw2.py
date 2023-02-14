# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:20:58 2023
@author: TongYu
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

def calculate_entropy(labels):
    classes,class_counts=np.unique(labels,return_counts=True)
    entropy_value=np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts))
        for i in range(len(classes))])
    return entropy_value
# if there is no majority in a lead node
def nodomain(y):
    temp=np.array(list(Counter(y).items()))
    m=max(temp[:,1])
    if np.where(temp[:,1]==m)[0].size !=1:
        return 1
    return 0
#class of node
class Node:
    def __init__(self, x, y, nodeid,featureid=0,splitvalue=0,datal=[],datar=[],leaf="False",parent=0,childl=0,childr=0,leaflabel=0):
        self.x = x 
        self.y = y
        self.nodeid=nodeid
        self.featureid = featureid
        self.splitvalue = splitvalue
        self.datal = datal
        self.datar = datar
        self.leaf=leaf
        self.parent=parent
        self.childl=childl
        self.childr=childr
        self.leaflabel=leaflabel
#class of tree
class tree():
    def __init__(self, x, y, nodes=[],num=0,preindex=0):
        self.x = x 
        self.y = y
        self.num=num
        self.nodes =nodes
        self.preindex =preindex
    def inforgain(self,y,countmaxtrix):
        hy=calculate_entropy(y)
        hys=0
        num_s=sum(countmaxtrix[:,0])
        for i in range(2):
            hys=hys+countmaxtrix[i,1]*countmaxtrix[i,0]/num_s
        return hy-hys

    def findbestsplit(self,x,y):
        splits=[]
        best=[]
        bestgain=0
        num_p=y.size
        for i in range(2):
            list1, list2 = (list(t) for t in zip(*sorted(zip(x[:,i], y))))
            for j in range(num_p-1):
                if list1[j]!=list1[j+1]:
                    en1=calculate_entropy(list2[0:j+1])
                    en2=calculate_entropy(list2[j+1:num_p])
                    count1=np.array([[j+1,en1],[num_p-(j+1),en2]])
                    #entropy of any candidates split
                    num1=num_p-(j+1)
                    num2=j+1
                    entropyS=(-num1/num_p)*np.log2(num1/num_p)+(-num2/num_p)*np.log2(num2/num_p)
                    #inforgain1
                    inforgain1=self.inforgain(list2,count1)

                    splits.append([i,list1[j+1],inforgain1,entropyS])
                    if inforgain1> bestgain:
                        bestgain=inforgain1
                        best=[i,list1[j+1],inforgain1,entropyS]
        return np.array(splits),best
    def isleaf(self,node):
        if node.x is None:
            return True,[],[]
        if node.x.shape[0] ==1:
            return True,[],[]
        splits,best=self.findbestsplit(node.x,node.y)
        if sum(splits[:,3])==0 :
            return True,[],[]
        if sum(splits[:,2]) ==0 :
            return True,[],[]
        return False,splits,best
    def leaflabel(self,node):
        ylist=list(node.y[:,0])
        # y_pre=list(nodes[i].y[:,0])
        y_pre=0
        if np.unique(ylist).size ==1:
            y_pre=node.y[:,0][0]
        else:
            y_pre=max(set(ylist), key = ylist.count)
        if nodomain(ylist) ==1:
            y_pre = 1    
        return int(y_pre)
    def splittree(self,node):
        # node=Node(self.x,self.y,self.num)
        self.nodes.append(node)
        node.leaf,splits,best=self.isleaf(node)
        # best=findbestsplit(x,y)
        # list1, list2 = (list(t) for t in zip(*sorted(zip(x[:,int(best[0])], y))))
        if node.leaf is True:
            node.leaflabel=self.leaflabel(node)
            return
        if node.leaf is False:
            index=np.where(node.x[:,best[0]]>=best[1])[0]
            index2=np.where(node.x[:,best[0]]<best[1])[0]
            node.featureid = best[0]
            node.splitvalue = best[1]
            node.datal = (node.x[index,:],node.y[index,:])
            node.datar = (node.x[index2,:],node.y[index2,:])
            self.num=self.num+1
            node.childl=self.num
            node1=Node(node.x[index,:],node.y[index,:],self.num)
            self.splittree(node1)
            self.num=self.num+1
            node.childr=self.num
            node2=Node(node.x[index2,:],node.y[index2,:],self.num)
            self.splittree(node2)
    # Make a prediction with a decision tree
    def predict(self,inputx):
        node=self.nodes[self.preindex]
        row=node.featureid
        value=node.splitvalue
        if node.leaf== True:
            self.preindex=0
            return node.leaflabel
        if inputx[row]>=value:
            self.preindex=node.childl
            return self.predict(inputx)
        else:
            self.preindex=node.childr
            return self.predict(inputx)
    
def printtree(nodes):
    num=len(nodes)
    text=''
    for i in range(num):
        if nodes[i].leaf==True:
            text=text+'node_id:'+str(nodes[i].nodeid)+' leaf. predict: '+str(nodes[i].leaflabel)+'\n'
        else:
            text=text+'node_id:'+str(nodes[i].nodeid)+' feature:'+str(nodes[i].featureid)+' threshold:'+str(nodes[i].splitvalue)+' leftchild:'+str(nodes[i].childl)+' rightchild:'+str(nodes[i].childr)+'\n'
    return text
def evaluate(tree,x,y):
    num=y.size
    acc=0
    for i in range(num):
        result=tree.predict(x[i,:])
        if result==y[i]:
            acc+=1
    return acc/num
def drawboundry(tree,x,y,n):
    xlist = np.linspace(min(x), max(x), n)
    ylist = np.linspace(min(y), max(y), n)
    X, Y = np.meshgrid(xlist, ylist)
    
    m=X.flatten()
    num=m.size
    m2=Y.flatten()
    mm=np.vstack((m,m2))
    Z=np.empty_like(X).flatten()
    for i in range(num):
        Z[i]=tree.predict(mm[:,i])
    Z=Z.reshape(X.shape)
    plt.contourf(X, Y, Z,cmap='summer')
# In[question 2]
x=np.array([[1,0],[0,1],[1,0],[0,1]])
y=np.array([1,0,0,1]).transpose()

colors=['g','r','r','g']

plt.scatter(x[:,0],x[:,1] , c=colors, alpha=0.5)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()
# In[question 3]
def inforgain(y,countmaxtrix):
    hy=calculate_entropy(y)
    hs=0
    num_s=sum(countmaxtrix[:,0])
    for i in range(2):
        hs=hs+countmaxtrix[i,1]*countmaxtrix[i,0]/num_s
    return hy-hs
with open('E:/UW/CS760/HW2/data/Druns.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])

splits=[]
best=[]
bestgain=0
num_p=y.size

for i in range(2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(x[:,i], y))))
    for j in range(num_p-1):
        if list1[j]!=list1[j+1]:
            en1=calculate_entropy(list2[0:j+1])
            en2=calculate_entropy(list2[j+1:num_p])
            count1=np.array([[j+1,en1],[num_p-(j+1),en2]])
            #entropy of any candidates split
            num1=num_p-(j+1)
            num2=j+1
            entropyS=(-num1/num_p)*np.log2(num1/num_p)+(-num2/num_p)*np.log2(num2/num_p)
            #inforgain1
            inforgain1=inforgain(list2,count1)

            splits.append([i,list1[j+1],inforgain1,entropyS])
            if inforgain1> bestgain:
                bestgain=inforgain1
                best=[i,list1[j+1],inforgain1,entropyS] 
# In[question 4]
with open('E:/UW/CS760/HW2/data/D3leaves.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])
root=Node(x,y,0)
dtree=tree(x,y)
dtree.splittree(root)
gragh=printtree(dtree.nodes)
print(gragh)
# In[question 5]
with open('E:/UW/CS760/HW2/data/D2.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])
root=Node(x,y,0)
dtree=tree(x,y)
dtree.splittree(root)
gragh=printtree(dtree.nodes)
print(gragh)
# In[question 6]
with open('E:/UW/CS760/HW2/data/D1.txt') as f:
    lines = f.readlines()
num1=len(lines)  
x1=np.zeros((num1,2))
y1=np.zeros((num1,1))
for i in range(num1):
    s=lines[i].split(' ')
    x1[i,0]=float(s[0])
    x1[i,1]=float(s[1])
    y1[i,0]=int(s[2])
index1=np.where(y1==0)[0]
index2=np.where(y1==1)
plt.scatter(x1[index1,0],x1[index1,1],c='b')
plt.scatter(x1[index2,0],x1[index2,1],c='r')
#boundary line
plt.plot([0, 1], [0.201829, 0.201829], color='green', linestyle='-', linewidth=5)
plt.show()

with open('E:/UW/CS760/HW2/data/D2.txt') as f:
    lines = f.readlines()
num2=len(lines)  
x2=np.zeros((num2,2))
y2=np.zeros((num2,1))
for i in range(num2):
    s=lines[i].split(' ')
    x2[i,0]=float(s[0])
    x2[i,1]=float(s[1])
    y2[i,0]=int(s[2])

root=Node(x2,y2,0)
dtree=tree(x2,y2)
dtree.splittree(root)
#boundary line
drawboundry(dtree,x2[:,0],x2[:,1],100)
index1=np.where(y2==0)[0]
index2=np.where(y2==1)
plt.scatter(x2[index1,0],x2[index1,1],marker="^",facecolors='none', edgecolors='k')
plt.scatter(x2[index2,0],x2[index2,1],marker="o",facecolors='none', edgecolors='k')
plt.show()
# In[question 7_1]
with open('E:/UW/CS760/HW2/data/Dbig.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])
    
def split_train_test(mylist,train_num):
    mylist=list(mylist)
    random.Random(1).shuffle(mylist)
    train_set=mylist[0:train_num]
    test_set=mylist[train_num:]
    return train_set,test_set

trainset5,testset5=split_train_test(np.hstack((x, y)),8192)

trainset4=trainset5[0:2048]
testset4=testset5
testset4=testset4+trainset5[2048:]

trainset3=trainset4[0:512]
testset3=testset4
testset3=testset3+trainset4[512:]

trainset2=trainset3[0:128]
testset2=testset3
testset2=testset2+trainset3[128:]

trainset1=trainset2[0:32]
testset1=testset2
testset1=testset1+trainset2[32:]

train=np.array(trainset5)
test=np.array(testset5)

x=train[:,0:2]
y=train[:,2:3]
root=Node(x,y,0)
dtree=tree(x,y)
dtree.splittree(root)
gragh=printtree(dtree.nodes)
print(gragh)
acc=evaluate(dtree,test[:,0:2],test[:,2])
err=1-acc
# In[question 7_2]
xplot=np.array([32,128,512,2048,8012])
yplot=np.array([0.0912,0.0674,0.0485,0.0265,0.0183])
# plt.scatter(xplot,yplot,)
plt.plot(xplot, yplot, color='green', linestyle='-', linewidth=5)
# In[question 7_3]
drawboundry(dtree,test[:,0],test[:,1],300)
plt.show()
# In[sklearn]
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

with open('E:/UW/CS760/HW2/data/Dbig.txt') as f:
    lines = f.readlines()
num=len(lines)  
x=np.zeros((num,2))
y=np.zeros((num,1))
for i in range(num):
    s=lines[i].split(' ')
    x[i,0]=float(s[0])
    x[i,1]=float(s[1])
    y[i,0]=int(s[2])
    
def split_train_test(mylist,train_num):
    mylist=list(mylist)
    # random.shuffle(mylist)
    random.Random(1).shuffle(mylist)
    train_set=mylist[0:train_num]
    test_set=mylist[train_num:]
    return train_set,test_set

trainset5,testset5=split_train_test(np.hstack((x, y)),8192)

trainset4=trainset5[0:2048]
testset4=testset5
testset4=testset4+trainset5[2048:]

trainset3=trainset4[0:512]
testset3=testset4
testset3=testset3+trainset4[512:]

trainset2=trainset3[0:128]
testset2=testset3
testset2=testset2+trainset3[128:]

trainset1=trainset2[0:32]
testset1=testset2
testset1=testset1+trainset2[32:]

train=np.array(trainset1)
test=np.array(testset1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(train[:,0:2],train[:,2:3])

#Predict the response for test dataset
y_pred = clf.predict(test[:,0:2])
acc=accuracy_score(test[:,2:3], y_pred)
print(clf.tree_.node_count)
print(1-acc)

xplot=np.array([32,128,512,2048,8192])
yplot=np.array([0.1949,0.0529,0.0486,0.0294,0.0160])
plt.plot(xplot, yplot, color='green', linestyle='-', linewidth=5)
# In[Lagrange Interpolation]
import math
from scipy.interpolate import lagrange
from scipy.stats import describe

xlist = np.linspace(0, 1, 100)
ylist=np.empty_like(xlist)
for i in range(xlist.size):
    ylist[i]=math.sin(xlist[i])
    
xtest=np.random.random(100)
ytest=np.empty_like(xtest)
for i in range(xtest.size):
    ytest[i]=math.sin(xtest[i])
    
poly = lagrange(xlist, ylist)

def accuracy_test(name, y1, y2):
    stats = describe(np.abs(y1 - y2))
    print('Error for {}'.format(name))
    print('   mean:     {}'.format(stats.mean))
    
print('--- Interpolation accuracy test ---')
accuracy_test('train', ylist, poly(xlist))
accuracy_test('test', ytest, poly(xtest))
print()

# add zero mean gaussian noise python
noise = np.random.normal(0,100,100)
xlistnew=xlist+noise
poly = lagrange(xlist, ylist)

print('--- Interpolation accuracy test ---')
accuracy_test('train', ylist, poly(xlistnew))
accuracy_test('test', ytest, poly(xtest))
print()