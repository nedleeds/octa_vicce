import numpy as np
import numpy.random as random

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def sampling(p):
    print(f'type(test_imgs) :',type(p))
    print(f'np.shape(test_imgs) :',np.shape(p))
    l,r,c,k = np.shape(p)
    z = random.rand(l,r,c,k)    
    output = sigmoid(np.log(z)-np.log(1-z)+np.log(p)-np.log(1-p))
    return output

