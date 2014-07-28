'''
'''

import numpy as np
from bnpy.data import WordsData, BagOfWordsMinibatchIterator
import Bars2D

# FIXED DATA GENERATION PARAMS
K = 50 # Number of topics
V = 2500 # Vocabulary Size
SEED = 8675309

Defaults = dict()
Defaults['seed'] = SEED
Defaults['nDocTotal'] = 2000
Defaults['nWordsPerDoc'] = 2 * V / (K/2)

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['beta'] = trueBeta

# TOPIC by WORD distribution
PRNG = np.random.RandomState(SEED)
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG)

def get_data_info():
  s = 'Toy Bars Data with %d true topics. Each doc uses ONE topic.' % (K)
  return s

def get_data(**kwargs):
    ''' Create and return dataset.

        Keyword Args
        -------
        nDocTotal
        nWordsPerDoc
    '''
    updateKwArgsWithDefaults(kwargs)
    Data = WordsData.CreateToyDataFromMixModel(**kwargs)
    Data.summary = get_data_info()
    return Data

def get_minibatch_iterator(**kwargs):
    ''' Create dataset and iterator to traverse that dataset in minibatches.

        Keyword Args
        -------
        nDocTotal
        nWordsPerDoc
    '''
    Data = get_data(**kwargs)
    DataIterator = BagOfWordsMinibatchIterator(Data, **kwargs)
    return DataIterator


def updateKwArgsWithDefaults(kwargs):
  for key in Defaults:
    if key not in kwargs:
      kwargs[key] = Defaults[key]

if __name__ == '__main__':
  import bnpy.viz.BarsViz
  WData = WordsData.CreateToyDataFromMixModel(**Defaults)
  bnpy.viz.BarsViz.plotExampleBarsDocs(WData)
