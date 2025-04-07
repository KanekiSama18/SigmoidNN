import numpy as np

#Activation Function: Sigmoid function

def sig(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#Input Dataset

x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#Output Dataset

out = np.array([[0,0,1,1]]).T

#Calculation: Seeding Random Numbers

np.random.seed(1)

#Initializing Weights: Random, With mean 0

syn = 2*np.random.random((3,1)) - 1

for i in range(10000):

    #Forward Propagation
    l0 = x
    l1 = sig(np.dot(l0, syn))

    #Error Calculation
    l1_err = out - l1

    l1_delta = l1_err * sig(l1,True)

    #Updating Weights
    syn += np.dot(l0.T,l1_delta)

print("Output: ", l1)