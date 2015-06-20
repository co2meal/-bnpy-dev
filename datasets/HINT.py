
import numpy as np
from bnpy.data import GraphXData
import os
import scipy.io


datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND HINT DATASET DIRECTORY:\n' + datasetdir)
'''
filepath = os.path.join(datasetdir, 'rawData', 'hprd.txt')
if not os.path.isfile(filepath):
  raise ValueError('CANNOT FIND HPRD DATASET TXT FILE:\n' + matfilepath)
'''

def get_data(matrixType='raw'):
  '''
  matrixType should be:
     'raw' : The raw, binary interaction network
     'rawHeat' : Heat diffusion matrix, assuming unit heat has been placed on
                 each of the nodes of the raw graph
     'exchangedHeat' : Heat exchanged with lots of heat placed on nodes 
                       biologically known to be associated with cancer
  '''
  global dataType
  dataType = matrixType
  print '*** HINT DATASET LOADING IN DATA TYPE ', matrixType, ' *****'
  if matrixType == 'raw':
    Data = GraphXData.LoadFromFile(os.path.join(datasetdir,
                                                'rawData',
                                                'HINTraw.txt'),
                                   isSparse=True)
  elif matrixType == 'rawHeat':
    Data = GraphXData.LoadFromFile(os.path.join(datasetdir,
                                                'rawData',
                                                'HINTrawHeat.mat'),
                                   isSparse=False)

  elif matrixType == 'heatExchange':
    Data = GraphXData.LoadFromFile(os.path.join(datasetdir,
                                                'rawData',
                                                'HINTheatExchange.mat'),
                                   isSparse=False)
    

    

  Data.summary = get_data_info()
  Data.name = get_short_name()
  return Data


def get_short_name():
  global dataType
  return 'HINT' + dataType

def get_data_info():
  return 'TODO TODO \n \n TODO TODO'


def preproc_raw_data():
  genes = np.genfromtxt(os.path.join(datasetdir,
                                     'rawData',
                                     'HINT',
                                     'inthint_index_genes'),
                        dtype='str')
  heat = np.genfromtxt(os.path.join(datasetdir,
                                     'rawData',
                                     'HINT',
                                     'new-pipeline-hint-freq-heat.txt'),
                        dtype='str')
  rawHeatMtx = scipy.io.loadmat(os.path.join(datasetdir,
                                     'rawData',
                                     'HINT',
                                     'inthint_ppr_0.45.mat'))
  rawHeatMtx = rawHeatMtx['PPR']
  raw = np.loadtxt(os.path.join(datasetdir,
                                'rawData',
                                'HINT',
                                'inthint_edge_list'),
                   dtype='int')
  raw -= 1 # make 0-indexed

  
  # Create indicies into the raw PPR (heat diffusion) matrix
  #    NOTE : will be 0-indexed
  inds = np.zeros(heat.shape[0])
  for i in xrange(heat.shape[0]):
    tmp =  np.where(heat[i,0] == genes[:,1])[0]
    inds[i] = np.where(heat[i,0] == genes[:,1])[0]


  # Pick out raw edges that are actually used
  edges = list()
  indset = set(inds.astype(int))
  E = raw.shape[0] # make zero indexed!
  for e in xrange(E):
    if raw[e,0] not in indset or raw[e,1] not in indset:
      continue
    edges.append((raw[e,0],raw[e,1]))
    edges.append((raw[e,1],raw[e,0]))


  # Make the raw heat matrix
  #rawHeatMtx = rawHeatMtx[inds,:]
  #rawHeatMtx = rawHeatMtx[:,inds]
  #scipy.io.savemat(os.path.join(datasetdir,
  #                              'rawData',
  #                              'HINTrawHeat.mat'),
  #                 {'X':rawHeat})
  rawHeatMtx = scipy.io.loadmat(os.path.join(datasetdir,
                                             'rawData',
                                             'HINTrawHeat.mat'))['X']

  # Make heat exchange matrix
  scores = heat[:,1]
  scores = np.asarray([float(s) for s in scores])
  D = np.diag(scores)
  exch = np.dot(rawHeatMtx, D)
  
  scipy.io.savemat(os.path.join(datasetdir,
                                'rawData',
                                'HINTheatExchange.mat'),
                   {'X':exch})
  scipy.io.savemat(os.path.join(datasetdir,
                                'rawData',
                                'HINT',
                                'inds.mat'),
                   {'inds':inds})
  np.savetxt(os.path.join(datasetdir, 'rawData', 'HINTraw.txt'),
             edges, fmt='%d')


if __name__ == '__main__':
  import networkx as nx
  import matplotlib.pyplot as plt
  
  Data = get_data(matrixType='raw')
  #colors = np.argmax(Data.TrueParams['pi'], axis=1)
  #relviz.plotTrueLabels('HINT', Data, gtypes=['Actual'])

  f,ax = plt.subplots(1)
  G = nx.Graph()
  for e in Data.edgeSet:
    G.add_edge(e[0], e[1])
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos)
  nx.draw_networkx_edges(G, pos)
  plt.show()
  from IPython import embed; embed()



#preproc_raw_data()

    
