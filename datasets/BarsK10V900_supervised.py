'''
BarsK10V900_supervised.py

Toy Bars data, with K=10 topics and vocabulary size 900.
5 horizontal bars, and 5 vertical bars. Horizontal bars give
no response, vertical bars give response.

#Generated via the standard LDA generative model
#  see WordsData.CreateToyDataFromLDAModel for details.
'''
import numpy as np
from bnpy.data import WordsData_slda
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 8  # Number of topics
V = 100  # Vocabulary Size
gamma = 0.5  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 1000
Defaults['nWordsPerDoc'] = 2 * V / (K / 2)

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'], Defaults['eta'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG,slda=True)


def get_data_info():
    s = 'Toy Bars Data for SLDA with %d true topics. Each doc uses 1-3 bars.\n' % (K)
    s += 'Each horizontal bar has no response, each vertical bar has integer response -2, -1, 0, 1, 2.'
    return s

def get_data(seed=SEED, **kwargs):
    ''' Create toy dataset using bars topics.

    Keyword Args
    ------------
    seed : int
        Determines pseudo-random generator used to make the toy data.
    nDocTotal : int
        Number of total documents to create.
    nWordsPerDoc : int
        Number of total words to create in each document (all docs same length)
    '''
    Data = CreateToyDataFromSLDAModel(seed=seed, **kwargs)
    Data.name = 'BarsK10V900'
    Data.summary = get_data_info()
    return Data

def get_test_data(seed=6789, nDocTotal=50, **kwargs):
    ''' Create dataset of "heldout" docs, for testing purposes.

    Uses different random seed than get_data, but otherwise similar.
    '''
    Data = CreateToyDataFromSLDAModel(seed=seed, nDocTotal=nDocTotal, **kwargs)
    Data.name = 'BarsK10V900'
    Data.summary = get_data_info()
    return Data


def CreateToyDataFromLDAModel(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    return WordsData.CreateToyDataFromLDAModel(**kwargs)

def CreateToyDataFromSLDAModel(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    kwargs['delta'] = 1
    return WordsData.CreateToyDataFromSLDAModel(**kwargs)


def showExampleDocs(pylab=None):
    import bnpy.viz.BarsViz as BarsViz
    WData = CreateToyDataFromLDAModel(seed=SEED)
    if pylab is not None:
        BarsViz.pylab = pylab
    BarsViz.plotExampleBarsDocs(WData)

if __name__ == '__main__':
    showExampleDocs()
