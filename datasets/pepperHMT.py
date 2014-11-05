'''
pepperHMT.py

A module to create HMT data from black and white pepper image
'''
import numpy as np
import matplotlib.pyplot as plt
from bnpy.data.QuadTreeData import QuadTreeData
from bnpy.distr.GaussDistr import GaussDistr
from bnpy.allocmodel.tree import HMTUtil
from scipy.io import loadmat

path = "/home/mterzihan/Desktop/denoising/pepper.mat"

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  d = loadmat(path)
  X = d['pyramid'][1]
  X = np.vstack(X)
  Data = QuadTreeData(X=X, nTrees=1)
  return Data

def get_data_info():
  return 'HMT data produced from black and white pepper image'

def get_short_name( ):
  ''' Return short string used in filepaths to store solutions
  '''
  return 'pepperHMT'