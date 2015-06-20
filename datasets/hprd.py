'''
hprd.py

TODO : EXPLANATION
'''

import numpy as np
from bnpy.data import GraphXData
import os


datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND HPRD DATASET DIRECTORY:\n' + datasetdir)

filepath = os.path.join(datasetdir, 'rawData', 'hprd.txt')
if not os.path.isfile(filepath):
  raise ValueError('CANNOT FIND HPRD DATASET TXT FILE:\n' + matfilepath)

########################################################### User-facing 
###########################################################
def get_data():
  Data = GraphXData.LoadFromFile(filepath, isSparse=True)
  Data.summary = get_data_info()
  Data.name = get_short_name()
  return Data

def get_short_name():
  return 'hprd'
  

def get_data_info():
  return 'Interaction network of 9205 human proteins with 36720 edges corresponding to biological interactions between proteins'
    

def preproc_rawData():
  '''
  The raw data taken from https://github.com/raphael-group/hotnet2 has an extra
  column in it.
  '''
  rawFilepath = os.path.join(datasetdir,
                             'rawData',
                             'hprd',
                             'hprd_edge_list')
  txt = np.loadtxt(rawFilepath)
  txt = txt.astype(int)
  
  # Edges are undirected, but raw data only lists one directed edge
  #   (i,j) instead of both (i,j) and (j,i)
  E = txt.shape[0]
  edges = np.zeros((E*2,2), dtype=np.int32)
  edges[:E,:] = txt[:,0:2]
  for e in xrange(E):
    edges[E+e,0] = edges[e,1]
    edges[E+e,1] = edges[e,0]
  np.savetxt(os.path.join(datasetdir, 'rawData', 'hprd.txt'),
             edges, fmt='%d')
  
  




if __name__ == '__main__':
  preproc_rawData()
  '''
  import networkx as nx
  import matplotlib.pyplot as plt
  import bnpy.viz.RelationalViz as relviz
  Data = get_data()
  fig, ax = plt.subplots(1)
  relviz.drawGraph(Data, curAx=ax, fig=fig)
  print 'ALL DONE'
  plt.show()
  '''
