'''
BarsK10V900.py

Toy Bars data, with K=10 topics and vocabulary size 900.
5 horizontal bars, and 5 vertical bars.

Generated via the standard LDA generative model
  see WordsData.CreateToyDataFromLDAModel for details.
'''
import numpy as np
from bnpy.data import WordsData
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 10 # Number of topics
V = 900 # Vocabulary Size
gamma = 0.5 # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 2000
Defaults['nWordsPerDoc'] = 2 * V / (K/2)

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG)

def get_data_info():
  s = 'Toy Bars Data with %d true topics. Each doc uses 1-3 bars.' % (K)
  return s

def get_data(seed=SEED, **kwargs):
    ''' 
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = CreateToyDataFromLDAModel(seed=seed, **kwargs)
    Data.summary = get_data_info()
    return Data

def CreateToyDataFromLDAModel(**kwargs):
  for key in Defaults:
    if key not in kwargs:
      kwargs[key] = Defaults[key]
  return WordsData.CreateToyDataFromLDAModel(**kwargs)

def showExampleDocs(pylab=None):
  import bnpy.viz.BarsViz as BarsViz
  WData = CreateToyDataFromLDAModel(seed=SEED)
  if pylab is not None:
    BarsViz.pylab = pylab
  BarsViz.plotExampleBarsDocs(WData)

if __name__ == '__main__':
  showExampleDocs()
