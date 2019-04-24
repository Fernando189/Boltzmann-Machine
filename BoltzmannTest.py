# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:26:35 2018

@author: Fernando Nuñez

Program to test the learning of the Boltzmann machine
"""
from random import randint
import numpy as np
import math
import random
from scipy.special import comb

W = np.genfromtxt('W.csv', delimiter=',')
W = np.reshape(W,(44,44))

# Function that creates V1
def getV1(length, numActive):
    V1 = [1]*numActive + [0]*(length-numActive)
    return random.sample(V1,len(V1))

# Function that returns the random shift, V3, and V2 after the shift
def getV2(V1):
    V3 = [0]*3
    x = randint(0,2)
    if x==0:
        V2 = V1[1:] + [V1[0]]
        V3[x] = 1
        return V3 , V2 # Left shift
    if x==1:
        V3[x] = 1
        return V3 , V1 # No shift
    if x==2:
        V3[x] = 1
        V2 = [V1[-1]] + V1[:-1]
        return V3 , V2 # Right shift
    
def getV(inputsV1,inputsV2,outputsV3,numHidden):
    V = inputsV1 + inputsV2 + [0]*3 + [0]*numHidden + [1]
    return V
    
def makeV(numActive):
    V1 = getV1(length,numActive)
    V3, V2 = getV2(V1)
    V = getV(V1,V2,V3,numHidden,length)
    return V

# Takes in a weight matrix, the random vector V, the vector of indices to be 
# probed K, the vector of random numbers to compare pk to F, the number of 
# unclamped units and a counter to traverse through K,F.
def runNetwork(W,V,K,F,T,numUnclamped): 
    """ TODO:Update h instead of V like in Navas code """
    counter = 0
    pV = np.zeros(3)
    # Loop through the temps twice
    for j in range(len(T)):
            # 2 'iterations' at each temperature
            for k in range(numUnclamped*2):
                # Iteration
                h = np.dot(W[K[counter],:],V)
                pk = 1/(1+math.exp(-h/T[j]))
                if F[counter] < pk:
                    V[K[counter]] = 1
                else:
                    V[K[counter]] = 0
                counter += 1
                pV = np.add(pV,V[16:19])
    return pV, V
 
# Length of V1 and V2
length = 8
# Number of different outputs for a single input
numOutputs = 3
# Number of hidden units
numHidden = 24
# +1 because of the threshold
size = 2*length+numOutputs+numHidden+1

""" I will unclamp only the units of V3 and H in order to learn the 'shift' """
""" V1, V2, threshold will be clamped """
numUnclamped = numOutputs+numHidden
# Initialize the temperature
T = [40,40,35,35,30,30,25,25,20,20,15,15,12,12,10,10,5,5,5,5,5,5,5,5,5,5]
# Number of random numbers to be used in testing
numRands = 1404  
numActive = [2,3,4,5,6]
# Number of possible combinations of a certain number of active units times 2

for z in range(len(numActive)):
    act = numActive[z]
    numSweeps = math.floor(comb(length,act))
    error = np.zeros((len(numActive),numSweeps))
    for s in range(numSweeps):
        np.random.seed(s+numSweeps)
        fUnc = 2*length
        lUnc = size-1
        luckyT = [randint(fUnc,lUnc) for i in range(numRands)]
        luckyT2 = [random.uniform(0,1) for i in range(numRands)]
        # Initialize V for test
        V1 = getV1(length,act)
        V3, V2 = getV2(V1)
        V = getV(V1,V2,V3,numHidden)
        # Get the probabilistic vector of the shift units V3
        pV, Vf = runNetwork(W,V,luckyT,luckyT2,T,numUnclamped)
        if np.argmax(pV) != np.argmax(V3):
            error[z][s] += 1
    accuracy = 1-np.mean(error[z,:])
    print('accuracy = ',math.floor(accuracy*100),'% with ', act, 'active units')
