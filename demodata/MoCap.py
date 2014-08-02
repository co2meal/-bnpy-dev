'''
MoCap.py

Dataset generated by motion capture of humans performing various exercises
There are 10 exercises, and (x,y) coordinates of 6 different joints are observed
  at each timestep.
'''

import numpy as np
from bnpy.data import SeqXData, MinibatchIterator
import readline


def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=3, nObsBatch=2, nObsTotal=25000, nLap=1, startLap=0, **kwargs):
  '''
    Args
    --------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    dataorderseed : integer seed that determines
                     (a) how data is divided into minibatches
                     (b) order these minibatches are traversed

   Returns
    -------
      bnpy MinibatchIterator object, with nObsTotal observations
        divided into nBatch batches
  '''
  X, fullZ, seqInds = get_XZ()
  Data = SeqXData(X = X, TrueZ = fullZ, seqInds = seqInds)
  Data.summary = get_data_info()
  DataIterator = MinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, \
                                       nLap=nLap, startLap=startLap, \
                                       dataorderseed=dataorderseed)
  return DataIterator
    
def get_XZ():
    X = list()
    Z = list()
    zTrue = open('/home/will/bnpy/bnpy-dev/demodata/mocap6/zTrue.dat', 'r')
    seqs = open('/home/will/bnpy/bnpy-dev/demodata/mocap6/SeqNames.txt', 'r')

    for line in seqs:
      line = line[:-1] #eat the \n at the end of each line
      file = open('/home/will/bnpy/bnpy-dev/demodata/mocap6/'+line+'.dat', 'r')
      seqX = list()
      
      seqZ = zTrue.readline()
      seqZ = seqZ.split(' ') 
      seqZ = seqZ[:-1] 
      seqZ = [int(i) for i in seqZ]
      Z.append(seqZ)

        #Read off all the observed joint positions
      for dataPoint in file:
        dataPoint = dataPoint.split(' ')
        dataPoint = dataPoint[:-1]
        dataPoint = [float(i) for i in dataPoint]
        seqX.append(dataPoint)
      X.append(seqX)
        
      seqInds = np.array([0])
      fullZ = []
      for i in xrange(len(Z)):
        seqInds = np.append(seqInds, len(Z[i]) + seqInds[i])
        fullZ = np.append(fullZ, Z[i])
    X = np.vstack(X)

    return X, fullZ, seqInds


def get_data_info():
    return 'Multiple sequences of data from motion capture of humans performing exercises'

def get_short_name():
    return 'MoCap'

def get_data(**kwargs):
    X, fullZ, seqInds = get_XZ()
    Data = SeqXData(X = X, seqInds = seqInds, TrueZ = fullZ)
    Data.summary = get_data_info()
    return Data
            
            