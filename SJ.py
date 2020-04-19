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
class loss:
    def ms(tru,fed):
        return 0.5*np.sum((fed-tru)**2)
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
    def __init__(self,sizes,ders,acts,los):
        self.size=sizes
        self.derx=ders
        self.actx=acts
        self.losx=los
        self.w=np.random.rand((len(sizes)-1,max(sizes),len(sizes)-1,max(sizes)))
        self.b=np.random.rand((len(sizes)-1,max(sizes),len(sizes)-1,max(sizes)))

    def feedforward(self,input):#前向传播
         ass=input.gets()
         a=[[]]
         a.append([])
         for i in ass:
             a[0].append(i)
         for i in range(len(self.size)):
             for j in range(self.size[i+1]):
                 for js in range(self.size[i]):
                     a[i+1].append(self.actx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j]))
         return a[len(a)-1]
     #def op(input,out,eta):