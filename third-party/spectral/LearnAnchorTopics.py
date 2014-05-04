import os
import sys
import numpy as np
import scipy.sparse as sparse
import scipy.io

from fastRecover import do_recovery
from anchors import findAnchors
from Q_matrix import generate_Q_matrix 

root = os.path.abspath(__file__)
root = os.path.sep.join(root.split(os.path.sep)[:-1])
settings_file = os.path.join(root, 'settings.conf')

def run(DocWordMat, K=10, loss='L2', seed=0, settings_file=settings_file):
  '''
    Args
    -------
    DocWordMat : sparse CSR-matrix, size D x V

    Returns
    -------
    topics : numpy 2D array, size K x V

  '''
  params = Params(settings_file, seed=seed)

  if type(DocWordMat) == str:
    DocWordMat = scipy.io.loadmat(DocWordMat)['M']
  
  if not str(type(DocWordMat)).count('csr_matrix') > 0:
    raise NotImplementedError('Need CSR matrix')  
    
  Q = generate_Q_matrix(DocWordMat.T)
  anchors = selectAnchorWords(DocWordMat.tocsc(), Q, K, params)
  topics, topic_likelihoods = do_recovery(Q, anchors, loss, params) 
  return topics.T


def selectAnchorWords(DocWordMat, Q, K, params):
  
  if not str(type(DocWordMat)).count('csc_matrix') > 0:
    raise NotImplementedError('Need CSC matrix')  
    
  nDocsPerWord = np.diff(DocWordMat.indptr)
  candidateWords = np.flatnonzero(nDocsPerWord > params.anchor_thresh)

  anchors = findAnchors(Q, K, params, candidateWords.tolist())
  return anchors


def print_topics(topics, vocab_file, Ntop=10, anchors=None):
  if vocab_file.count('bnpy') > 0:
    MatDict = scipy.io.loadmat(vocab_file, squeeze_me=True)
    vocab = [str(word) for word in MatDict['vocab_dict']]
  else:
    vocab = file(vocab_file).read().strip().split()
  K = topics.shape[1]
  for k in xrange(K):
    topwords = np.argsort(-1 * topics[:, k])[:Ntop]
    if anchors is not None:
      print vocab[anchors[k]], ':',
    print ' '.join([vocab[w] for w in topwords])
    print ' '.join(['%.3f' % (topics[w,k]) for w in topwords])
    print ' '
    


class Params:

    def __init__(self, filename, seed=0):
        self.log_prefix=None
        self.checkpoint_prefix=None
        self.seed = seed

        for l in file(filename):
            if l == "\n" or l[0] == "#":
                continue
            l = l.strip()
            l = l.split('=')
            if l[0] == "log_prefix":
                self.log_prefix = l[1]
            elif l[0] == "max_threads":
                self.max_threads = int(l[1])
            elif l[0] == "eps":
                self.eps = float(l[1])
            elif l[0] == "checkpoint_prefix":
                self.checkpoint_prefix = l[1]
            elif l[0] == "new_dim":
                self.new_dim = int(l[1])
            elif l[0] == "seed":
                self.seed = int(l[1])
            elif l[0] == "anchor_thresh":
                self.anchor_thresh = int(l[1])
            elif l[0] == "top_words":
                self.top_words = int(l[1])
