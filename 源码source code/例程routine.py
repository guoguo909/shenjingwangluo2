import numpy as np
import network
net=network.NE([1,1,10,1],network.act.elu,network.der.elu)
for i in range(1000):
    net.AUSGD(np.array([[2,3],[3,4]]),1,1)
    print(net.los(np.array([2]),np.array([3])))
print(net.feedforward(np.array([2])))
