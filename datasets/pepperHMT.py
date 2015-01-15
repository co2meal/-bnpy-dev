'''
pepperHMT.py

A module to create HMT data from black and white pepper image
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.distr.GaussDistr import GaussDistr
from bnpy.allocmodel.tree import HMTUtil
from scipy.io import loadmat
from collections import deque

path = '/Users/mertterzihan/Desktop/pepper256.mat'

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  q = deque()
  d = loadmat(path)
  datass_graph = np.zeros(d['datass_graph'].shape)
  datass_graph[:] = d['datass_graph']
  children_matrix = np.zeros(d['children_matrix'].shape)
  children_matrix[:] = d['children_matrix']
  graph = np.zeros([datass_graph.shape[1], datass_graph.shape[0]])
  idx = 0
  doc_range = np.zeros([401, 1])
  for ii in xrange(400):
		q.append(ii)
		while len(q) > 0:
			curr = q.popleft()
			graph[idx,:] = datass_graph[:,curr]
			idx += 1
			ch_idx = children_matrix[curr, :]-1
			if ~(ch_idx==-1).any():
				q.extend(ch_idx.tolist())
		doc_range[ii+1] = idx
  return GroupXData(graph, doc_range)

def get_data_info():
  return 'HMT data produced from black and white pepper image'

def get_short_name( ):
  ''' Return short string used in filepaths to store solutions
  '''
  return 'pepperHMT'