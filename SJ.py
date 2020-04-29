import numpy as np
import math
import random
import h5py
A=0.01
class act:#激活函数
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))
    def tanh(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def none(x):
        return x
    def relu(x):
        return 1 * (x > 0) * x
    def leaky_relu(x):
        return np.where(x <=0, x*A, x)
    def elu(x):
        return np.where(x <=0, A*(np.exp(x)-1), x)
class der:#导数
    def sigmoid(x):
        return act.sigmoid(x)*(1-act.sigmoid(x))
    def tanh(x):
        return 1-math.pow(act.tanh(x),2)
    def none(x):
        return 1
    def relu(x):
        return 1 * (x > 0) * 1
    def leaky_relu(x):
        return np.where(x >0, 1, A)
    def elu(x):
        return np.where(x <=0, A*np.exp(x), 1)
class loss:#损失函数
    def ms(tru,fed):
        return 0.5*np.sum((fed-tru)**2)
    def square(tru,fed):
        return np.sum((fed-tru)**2)
class tensor:
    def __init__(self,x:np):
        self.a=x
    def lens(self):
        return len(self.a)
    def set(self,x,y):
        self.a[x]=y
    def get(self,x):
        return self.a[x]
    def gets(self):
        return self.a
class bp:
    def __init__(self,sizes,insizes=1,ders=der.none,acts=act.none,los=loss.ms):#定义
        self.size=sizes
        self.derx=ders
        self.actx=acts
        self.losx=los
        self.w=np.random.rand(len(sizes),max(sizes),len(sizes),max(sizes),insizes)
        self.b=np.random.rand(len(sizes),max(sizes),len(sizes),max(sizes),insizes)
        self.insize=insizes
    def getw(self,x,y,xx,yy):#获取权重
        return self.w[x,y,xx,yy]
    def getb(self,x,y,xx,yy):#获取偏置
        return self.b[x,y,xx,yy]
    def getws(self):#获取权重数组
        return self.w
    def getbs(self):#获取偏置数组
        return self.b
    def los(self,input,out):#损失
        a=[]
        for i in out:
            a.append(i.gets())
        a=np.array(a)
        return self.losx(a,self.feedforward(input))
    def feedforward(self,input):#前向传播
         ass=input
         a=[[]]
         ab=[[]]
         a.append([])
         ab.append([])
         for i in ass:
             a[0].append(i.gets())
             ab[0].append(i.gets())
         
         for i in range(len(self.size)-1):
             a.append([])
             ab.append([])
             for j in range(self.size[i+1]):
                 jj=np.array(0)
                 js=np.array(0)
                 for js in range(self.size[i]):
                     jj=jj+self.actx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j])
                     js=js+self.derx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j])
                 a[i+1].append(jj)
                 ab[i+1].append(js)
         self.feed=np.array(a)
         self.feeds=np.array(ab)
         return np.array(a[len(a)-2])
    def feedforwards(self,input):#前向传播多输出版本
         ass=input
         a=[[]]
         ab=[[]]
         a.append([])
         ab.append([])
         for i in ass:
             a[0].append(i.gets())
             ab[0].append(i.gets())
         
         for i in range(len(self.size)-1):
             a.append([])
             ab.append([])
             for j in range(self.size[i+1]):
                 jj=np.array(0)
                 js=np.array(0)
                 for js in range(self.size[i]):
                     jj=jj+self.actx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j])
                     js=js+self.derx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j])
                 a[i+1].append(jj)
                 ab[i+1].append(js)
         self.feed=np.array(a)
         self.feeds=np.array(ab)
         ret=[]
         for i in a[len(a)-2]:
             ret.append(tensor(np.array(i)))
         return ret
    def op(self,input,out,eta):#优化
        self.feedforward(input)
        self.updatew(input,self.back(out),eta)
    def back(self,out):#反向传播
        b=np.zeros((len(self.size),max(self.size),self.insize))
        out=np.array(out)
        ij=len(self.size)-1
        ib=0
        for i in out:
            a1=np.array(self.feed[len(self.feed)-2])
            a2=np.array(i.gets())
            a3=np.array(self.feeds[len(self.feed)-2])
            a5=-(a1-a2)*a3
            for ix in range(len(a5)):
                b[0][ix]=a5[ix][0]
            ib=ib+1
        for i in range(self.size[len(self.size)-1]):
            for j in range(self.size[len(self.size)-2]):
                b[1][i+j-1]=b[0][i]*self.w[len(self.w)-1][j][len(self.w)-2][i]+self.b[len(self.w)-1][j][len(self.w)-2][i]
        for i in range(len(self.size)-1):
            for j in range(self.size[ij-i]*self.size[ij-i-1]):
                b1=b[i+1][int(j/self.size[ij-i-1])]
                b3=self.feeds[ij-i][int(j/self.size[ij-i-1])]
                b[i+1][int(j/self.size[ij-i-1])]=np.array(b1*b3)
                for js in range(self.size[ij-i-1]):
                    b[i+1][js]=(b[i+1][int(j/self.size[ij-i-1])]*self.w[ij-i][js][ij-i-1][int(j/self.size[ij-i-1])-1]+self.b[ij-i][js][ij-i-1][int(j/self.size[ij-i-1])-1])
        return b
    def updatew(self,input,bs,eta):#更改权重偏置
        for i in range(len(self.size)-1):
            for j in range(self.size[i]):
                for js in range(self.size[i+1]):
                    self.w[i][j][i+1][js]=np.array(self.w[i][j][i+1][js])+np.array(self.feed[i][j]*bs[i+1][js]*eta)
                    self.b[i][j][i+1][js]=np.array(self.b[i][j][i+1][js])-np.array(eta*bs[i+1][js])
#