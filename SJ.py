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
    def square(tru,fed):
        return (fed-tru)**2
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
    def __init__(self,sizes,ders=der.none,acts=act.none,los=loss.ms):
        self.size=sizes
        self.derx=ders
        self.actx=acts
        self.losx=los
        self.w=np.random.rand(len(sizes),max(sizes)+1,len(sizes),max(sizes)+1)
        self.b=np.random.rand(len(sizes),max(sizes)+1,len(sizes),max(sizes)+1)
    def getw(self,x,y,xx,yy):
        return self.w[x,y,xx,yy]
    def getb(self,x,y,xx,yy):
        return self.b[x,y,xx,yy]
    def getws(self):
        return self.w
    def getbs(self):
        return self.b
    def los(self,input,out):
        a=[]
        for i in out:
            a.append(i.gets())
        a=np.array(a)
        return self.losx(a,self.feedforward(input))
    def feedforward(self,input):#前向传播
         ass=input
         a=[[]]
         a.append([])
         for i in ass:
             a[0].append(i.gets())
         for i in range(len(self.size)-1):
             a.append([])
             for j in range(self.size[i+1]):
                 for js in range(self.size[i]):
                     a[i+1].append(self.actx((a[i][js]*self.w[i][js][i+1][j])+self.b[i][js][i+1][j]))
         self.feed=a
         return np.array(a[len(a)-2])
    def op(self,input,out,eta):
        self.feedforward(input)
        self.back(self.feed,out)
    def back(self,a,out):
        b=[[]]
        b.append([])
        ij=len(self.sizes)-1
        b[0]=-(self.feed[len(self.feed)-2]-out)*self.feed[len(self.feed)-2]*(1-self.feed[len(self.feed)-2])
        b.append([])
        for i in range(self.sizes[len(self.sizes)-1]):
            for j in range(self.sizes[len(self.sizes)-2]):
                b[1].append(b[0][i]*self.w[len(self.w)-1][j][len(self.w)-2][i]+self.b[len(self.w)-1][j][len(self.w)-2][i])
        for i in range(len(self.sizes)-1):
            b.append([])
            for j in range(self.sizes[ij-i]):
                b[i+1][j]=self.feed[ij-i][j]*b[i+1,j]*(1-self.feed[ij-i][j])
                for js in range(self.sizes[ij-i-1]):
                    b.append([])
                    b[i+2].append(b[i+1][j]*self.w[ij-i][js][ij-i-1][j]+self.b[ij-i][js][ij-i-1][j])
        return b
#    def updatew(self,input,b,eta):

#
