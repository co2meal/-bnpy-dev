
'''
Relativity.py

Undirected network of co-authorship in general relativity and quantum cosmology
papers published on arXiv from January 1993 to April 2003.  The presence of an
edge indicates that two people published a paper together.  The graph contains
5242 nodes and 14,496 undirected edges.

For the raw dataset, see:
https://snap.stanford.edu/data/ca-GrQc.html
'''


import numpy as np
import scipy.io
import os

from bnpy.data import GraphXData

# Get path to the .mat file with the data
datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND RELATIVITY DATASET DIRECTORY:\n' + datasetdir)

filepath = os.path.join(datasetdir, 'rawData', 'relativity.txt')
if not os.path.isfile(filepath):
  raise ValueError('CANNOT FIND RELATIVITY DATASET TXT FILE:\n' + filepath)

  
########################################################### User-facing 
###########################################################
def get_data(**kwargs):
  Data = GraphXData.LoadFromFile(filepath, isSparse=True)
  Data.summary = get_data_info()
  Data.name = get_short_name()
  return Data

def get_data_info():
  return 'Co-authorship network of general relativity and quantum cosmology publications. Contains 5242 nodes and 14,496 undirected edges'

def get_short_name():
  return 'Relativity'


def preproc_raw_data():
  '''The raw ca-GrQc.txt file gives a graph with 355 connected components.
     This function writes rawData/relativity.txt which gives the largest of
     these.
  '''
  import networkx as nx
  rawFilepath = os.path.join(datasetdir, 'rawData', 'ca-GrQc.txt')
  if not os.path.isfile(filepath):
    raise ValueError('CANNOT FIND RELATIVITY DATASET MAT FILE:\n' + filepath)
    
  txt = np.loadtxt(rawFilepath)
  txt = txt.astype(int)
  G = nx.DiGraph()
  for i in xrange(txt.shape[0]):
    G.add_edge(txt[i,0], txt[i,1])    
  comps = nx.weakly_connected_component_subgraphs(G)
  edges = np.asarray(comps[0].edges())
  np.savetxt(os.path.join(datasetdir, 'rawData', 'relativity.txt'),
             edges.astype(int), fmt='%d')
  



if __name__ == '__main__':
  Data = get_data()
  import matplotlib.pyplot as plt
  from bnpy.viz import RelationalViz as relviz
  import networkx as nx

  f,ax = plt.subplots(1)
  G = nx.DiGraph()
  for e in Data.edgeSet:
    G.add_edge(e[0], e[1])
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos)
  nx.draw_networkx_edges(G, pos)
  plt.show()
  from IPython import embed; embed()
