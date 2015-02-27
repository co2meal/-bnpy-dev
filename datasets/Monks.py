'''
Monks.py

TODO : DESCRIPTION ... WHICH OF THE MATRICIES FROM THE FULL DATASET IS THIS??

The dataset was gathered during a perdio of political turmoil in the cloister. The true labels (TrueZ) reflect the "faction labels" of each monk: 0 = Young Turks (rebel group), 1 = Loyal Opposition (monks who followed tradition and remained loyal), 2 = Outcasts (Monks who were not accepted by either faction), 3 = Waverers (Monks who couldn't decide on a group).

Note the full version of this dataset contains more relationships than the ones here.  For the full dataset and more information, see:
http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm#sampson
'''

import numpy as np
import scipy.io
import os

from bnpy.data import GraphXData


# Get path to the .mat file with the data
datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND MONKS DATASET DIRECTORY:\n' + datasetdir)

matfilepath = os.path.join(datasetdir, 'rawData', 'Monks.mat')
if not os.path.isfile(matfilepath):
  raise ValueError('CANNOT FIND MONKS DATASET MAT FILE:\n' + matfilepath)


########################################################### User-facing 
###########################################################
def get_data(**kwargs):
  Data = GraphXData.read_from_mat(matfilepath)
  Data.summary = get_data_info()
  Data.name = get_short_name()
  return Data

def get_data_info():
  return '\n\n************** TODO TODO TODO ******************************\n\n.'

def get_short_name():
  return 'Monks'

  
if __name__ == '__main__':
  import networkx as nx
  import matplotlib.pyplot as plt
  
  adjMtx = scipy.io.loadmat(matfilepath)
  adjMtx = adjMtx['X'].reshape((18,18), order='F')
  G = nx.DiGraph(adjMtx)
  nx.draw_spring(G)
  plt.show()
