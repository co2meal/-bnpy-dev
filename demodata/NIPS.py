'''
NIPSCorpus.py

'''
from bnpy.data import WordsData
import os

datadir = '/Users/daeil/Dropbox/research/bnpy/data/'
NIPSmatfile = 'nips_bnpy.mat'
matfilepath = os.environ['BNPYDATADIR'] + NIPSmatfile

if not os.path.exists(matfilepath):
    matfilepath = datadir + NIPSmatfile

def get_data(**kwargs):
    ''' Grab data from matfile specified by matfilepath
    '''
    Data = WordsData.read_from_mat(matfilepath)
    Data.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return Data

def get_data_info(D, V):
    return 'NIPS bag-of-words data. D=%d. VocabSize=%d' % (D,V)
