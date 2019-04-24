# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:41:45 2019

@author: Fernando Nuñez

Thesis Proposal 1: Testing trained W with length of V1, V2 = 10

Note: There are three cases being tested here:
case 1: Unambiguous stimulus, case 2: Maximally ambiguous stimulus
case 3: Biased ambiguous stimulus, parameter p is number of units shifted left
To change the case, set the case variable in line 111 to 1,2, or 3.
If case = 3, also set the variable p to a value from 1 to 10
In all cases, test is performed for 2-8 active units
"""

from random import randint
import numpy as np
import math
import random
from scipy.special import comb

W = np.genfromtxt('W10.csv', delimiter=',')
W = np.reshape(W,(54,54))

# Function that creates V1
def getV1(length, numActive):
    V1 = [1]*numActive + [0]*(length-numActive)
    return random.sample(V1,len(V1))

def getV2(V1, case):
    if case == 1: # Unambiguous stimuli (True shift, each with p=1/3)
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
    if case == 2: # Maximally ambiguous stimuli (first half left, second half right)
        V2L = V1[1:] + [V1[0]]
        V2R = [V1[-1]] + V1[:-1]
        V2 = V2L[0:5]
        V2 = V2 + V2R[5:10]
        return V2
    if case == 3: # Probabilistic stimuli
        V2L = V1[1:] + [V1[0]]
        V2R = [V1[-1]] + V1[:-1]
        V2 = V2L[0:p]
        V2 = V2 + V2R[p:10]
        return V2
        return 0
        
    
def getV(inputsV1,inputsV2,numHidden):
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
                pV = np.add(pV,V[20:23])
    return pV, V
 
# Length of V1 and V2
length = 10
# Number of different outputs for a single input
numOutputs = 3
# Number of hidden units
numHidden = 30
# +1 because of the threshold
size = 2*length+numOutputs+numHidden+1

""" I will unclamp only the units of V3 and H in order to learn the 'shift' """
""" V1, V2, H, threshold will be clamped """
numUnclamped = numOutputs+numHidden
# Initialize the temperature
T = [40,40,35,35,30,30,25,25,20,20,15,15,12,12,10,10,5,5,5,5,5,5,5,5,5,5]
# Number of random numbers to be used in testing
numRands = 1755
numActive = [2,3,4,5,6,7,8]

""" case 1: Unambiguous stimulus, case 2: Maximally ambiguous stimulus """
""" case 3: Biased ambiguogu stimulus """
case = 3
""" For case = 3, p is the bias of the shift (number of units that go left) """
p = 8

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
        if case == 1:
            V3, V2 = getV2(V1, case)
            V = getV(V1,V2,numHidden)
            # Get the probabilistic vector of the shift units V3
            pV, Vf = runNetwork(W,V,luckyT,luckyT2,T,numUnclamped)
            if np.argmax(pV) != np.argmax(V3):
                error[z][s] += 1
        if case == 2:
            V2 = getV2(V1, case)
            V = getV(V1,V2,numHidden)
            # Get the probabilistic vector of the shift units V3
            pV, Vf = runNetwork(W,V,luckyT,luckyT2,T,numUnclamped)
            pV = pV/sum(pV)
            pV = [round(x,3) for x in pV]
        if case == 3:
            V2 = getV2(V1, case)
            V = getV(V1,V2,numHidden)
            # Get the probabilistic vector of the shift units V3
            pV, Vf = runNetwork(W,V,luckyT,luckyT2,T,numUnclamped)
            pV = pV/sum(pV)
            pV = [round(x,3) for x in pV]
    if case == 1:
        accuracy = 1-np.mean(error[z,:])
        print('accuracy = ',math.floor(accuracy*100),'% with ', act, 'unambiguous active units')
    if case == 2:
        print('Model converged to V3 = ',pV,' with ',act,'active MAS units')
    if case == 3:
        print('Model converged to V3 = ',pV,' with ',act,'active units and a bias of ',p,' units shifted left')
    