# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:47:04 2019

@author: Fernando Nuñez

Thesis Proposal 1: Using a Boltzmann machine to solve the 1-d shifter problem 
trained with length of V1, V2 = 10
Note that after running this code the trained W is saved as a csv file: W10.csv
and its values are rounded to 5 decimal places

"""

from random import randint
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

# Function that creates V1
def getV1(length):
    V1 = [randint(0,9) for i in range(length)]
    for i in range(length):
        if V1[i] <= 2:
            V1[i] = 1
        else:
            V1[i] = 0
    return V1

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

# Function that merges V1,V2,V3 and adds hidden units and threshold to create V
def getV(inV1,inV2,outV3,numH,numIn,length,index):
# Randomly select an input vector consisting of a V1,V2,V3 triplet
    lucky = index
# Collect all info into the 'lucky' (luckieth) vector and clamp its inputs
# Initialize all the connections as 0 of the hidden units and the threshold as 1
    V = [0]
    V[0:length-1] = inV1[lucky,:]
    V[length:(2*length)-1] = inV2[lucky,:]
    V[(2*length):(2*length)+2] = outV3[lucky,:]
    V[2*length+3:size-1] = [0]*numH
    V.append(1)
    return V

def makeV():
    # Container matrices for the vectors before shift, after shif and the shift itself
    inputsV1 = np.zeros((numInputs, length))
    inputsV2 = np.zeros((numInputs, length))
    outputsV3 = np.zeros((numInputs, numOutputs))
    # Loop that produces numInputs inputs and the corresponding 'target' outputs
    for i in range(numInputs):
        v1 = getV1(length)
        v3, v2 = getV2(v1)
        inputsV1[i,:] = v1
        inputsV2[i,:] = v2
        outputsV3[i,:] = v3
    return inputsV1, inputsV2, outputsV3 

# Function that returns an architecture matrix based on visible/hidden units' connection
def getArchM(size,fV,lV):
    archM = np.zeros((size,size))
    archM[0:fV,fV:lV] = 1
    archM[fV:lV,0:fV] = 1
    archM[lV,:] = 1
    archM[:,lV] = 1
    archM[lV,lV] = 0
    return archM

# Function that creates T, the initial weights matrix
def generateW(size,archM):
    # Generate skeleton for weights matrix
    W = 2*np.random.rand(size, size)-1
    # Generate random symmetric weights matrix with values in [-1,1]
    W0 = np.multiply(W,archM)
    # Initialize the thresholds
    W0[0:size-1,size-1] = [-1]
    W0[size-1,0:size-1] = [-1]
    return 0.5*(W0 + W0.T - np.diag(W0.diagonal()))

# Takes in a weight matrix, the random vector V, the vector of indices to be 
# probed K, the vector of random numbers to compare pk to F, the number of 
# unclamped units and a counter to traverse through K,F.
def runAnnealing(W,V,K,F,T,numUnclamped): 
    """ TODO:Update h instead of V like in Navas code """
    counter = 0
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
    return V, counter

def runRelax(W,V,K,F,T,numUnclamped,counter,P):
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
            # Sum the coocurrences every 2 iterations to P_ij
            """ Changed from kronecker to outer product """
            P  += np.outer(V,V)
     return P

# Length of V1 and V2
length = 10
# Number of inputs
numInputs = 20
# Number of different outputs for a single input
numOutputs = 3
# Number of hidden units
numHidden = 30
# Initialize the 30 hidden units as 0
hiddenUnits = [0]*numHidden
# +1 because of the threshold
size = 2*length+numOutputs+len(hiddenUnits)+1
# Get architecture matrix
archM = getArchM(size, 2*length+3, size-1)
# Get initial random symmetric weights matrix and copy it for debugging purposes
W = generateW(size,archM)
# initialize number of unclamped units, first and last unclamped indices for Phase+
numUnclamped = len(hiddenUnits)
fH = 2*length+3
lH = size-1
# Initialize the temperature
T = [40,40,35,35,30,30,25,25,20,20,15,15,12,12,10,10]
T10 = [10,10,10,10,10,10,10,10,10,10]
# Number of sweeps to be ran
numSweeps = 9000
# Start timer
t0 = time.clock()
numRandsP = 31200
numRandsM = 55120

for s in range(numSweeps):
    # Function that generates inputs
    inputsV1,inputsV2,outputsV3 = makeV()
    # Use different random seeds to obtain multiple sets of 20 in/outputs
    # The machine will learn from one input pattern at a time
    np.random.seed(s+numSweeps)
    # Initialize helper matrices
    numUnclamped = len(hiddenUnits)
    fUnc = 2*length+3
    lUnc = size-2
    DP = np.zeros((size,size))
    PPlus = np.zeros((size,size))
    luckyP = [randint(fUnc,lUnc) for i in range(numRandsP)]
    luckyP2 = [random.uniform(0,1) for i in range(numRandsP)]
    #Phase+
    for i in range(numInputs):
        # Initialize V for phase+
        V = getV(inputsV1,inputsV2,outputsV3,numHidden,numInputs,length,i)
        # 2 'iterations' for each temperature in annealing schedule
        V, counter = runAnnealing(W,V,luckyP,luckyP2,T,numUnclamped)
        PPlus = runRelax(W,V,luckyP,luckyP2,T10,numUnclamped,counter,PPlus)
    # Divide the resulting Pij+ by the number of units times the length of relaxation
    PPlus = (1/200)*PPlus
    
    #Phase-: same as Phase+ but unclamped units change
    numUnclamped = size-1
    fUnc = 0
    lUnc = size-2;
    PMinus = np.zeros((size,size))
    luckyM = [randint(fUnc,lUnc) for i in range(numRandsM)]
    luckyM2 = [random.uniform(0,1) for i in range(numRandsM)]
    for i in range(numInputs):
        V = getV(inputsV1,inputsV2,outputsV3,numHidden,numInputs,length,i)
        V, counter = runAnnealing(W,V,luckyM,luckyM2,T,numUnclamped)
        PMinus = runRelax(W,V,luckyM,luckyM2,T10,numUnclamped,counter,PMinus)
    PMinus = (1/200)*PMinus
        
    # End of sweep, update T
    DP = np.multiply(5*(PPlus - PMinus),archM)
    W = W + DP
    # Decay
    W = 0.9995*W
    # To get the avg time per sweep
t1 = time.clock()
# Stop timer 
    
print("time elapsed: ", round((t1-t0),2), " seconds")
print("avg time per sweep: ", round(((t1-t0)/numSweeps),2), " seconds")

# TODO: Update visualization tool to the one provided by Nava
fig, axes = plt.subplots(nrows=6, ncols=4)
i = 1
for ax in axes.flat:
    plot = np.zeros([4,10])
    plot[0,0] = W[22+i,-1]
    plot[0, 3:6] = W[22+i, 20:23]
    plot[2,:] = W[22+1, 0:10]
    plot[3, :] = W[22+i, 10:20]
    im = ax.imshow(plot, cmap='gray')
    i += 1
fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

W.tofile('W10.csv',sep=',',format='%10.5f')