import numpy as np
import math
import random
#E=2.71828
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
class loss:
    def ms(tru,fed):
        return 0.5*np.sum((fed-tru)**2)
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
class NE(object):
   
    def __init__(self, sizes:int,acts=act.none,ders=der.none,losss=loss.ms):#初始化用NE（）调用，self不是参数
        self.num_layers = len(sizes)
        self.sizes = sizes#大小
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#偏置
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]#权重
        self.actx=acts
        self.derx=ders
        self.OPres={"time":0,"loss":0.01}
        self.etab=0.00
        self.lossx=losss
        self.lo=float('inf')
        self.oi=True
    def feedforward(self, a):#前向传播
        for b, w in zip(self.biases, self.weights):
            a = self.actx(np.dot(w, a)+b)
        return a
    def getb(self,x):
        return self.biases[x]
    def getw(self,x):
        return self.weights[x]
    def setb(self,x,y):
        self.biases[x]=y
    def setw(self,x,y):
        self.weights[x]=y
    def getbs(self):
        return self.biases
    def getws(self):
        return self.weights
    def setbs(self,x):
        self.biases=x
    def setws(self,x):
        self.weights=x
    def los(self,input,out):
        return self.lossx(out,self.feedforward(input))
    def SGD(self, training_data, epochs, mini_batch_size, eta):
        test_data=None
        #self.etaa=eta
        eta=eta
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
   
        self.OPres["time"]=j
        self.OPres["loss"]=self.los(training_data[0,0],training_data[0,1])
    def AUSGD(self, training_data, epochs, mini_batch_size,aueta=0.000001):
        test_data=None
        #self.etaa=eta
        if self.oi:
            eta=self.etab+aueta
        else:
            eta=self.etab
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
        self.OPres["time"]=j
        self.OPres["loss"]=self.los(training_data[0,0],training_data[0,1])
        if self.oi:
             if self.lo>self.los(training_data[0,0],training_data[0,1]):
                 self.lo=self.los(training_data[0,0],training_data[0,1])
                 self.etab=eta
             else:
                self.oi=False
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y):#反向传播
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation =self.actx(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            self.derx(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp =self.derx(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    def save(self,filename):
        pn=[self.num_layers,self.etab,self.lo,self.oi]
        h5f = h5py.File(filename, 'w')
        for i in range(len(self.weights)):
                h5f.create_dataset("w"+str(i), data=self.weights[i])
        for i in range(len(self.biases)):
                h5f.create_dataset("b"+str(i), data=self.biases[i])
        h5f.create_dataset("pn", data=pn)
        h5f.create_dataset("sizes", data=self.sizes)
        siz=np.array([len(self.weights),len(self.biases)])
        h5f.create_dataset("siz", data=siz)
        h5f.close()
    def open(self,filename):
       h5f = h5py.File(filename, 'r')
       siz=h5f["siz"][:]
       self.weights=[]
       self.biases=[]
       sizess=[]
       for i in range(siz[0]):
            self.weights.append(h5f["w"+str(i)][:])
       for i in range(siz[1]):
            self.biases.append(h5f["b"+str(i)][:])
       pn=h5f["pn"][:]
       sizess.append(h5f["sizes"][:])
       h5f.close()
       [self.num_layers,self.etab,self.lo,self.oi]=pn
       self.sizes=np.array(sizess)
        