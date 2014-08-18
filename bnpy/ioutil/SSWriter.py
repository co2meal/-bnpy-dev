'''
'''
import numpy as np
import scipy.io
import os
from distutils.dir_util import mkpath

def save_counts(SS, fpath, prefix):
  ''' Save topic assignment counts from SuffStatBag to mat file persistently
      
      Args
      --------
      SS: SuffStatBag to read counts from
      fpath: absolute full path of directory to save in
      prefix: prefix for file name, like 'Iter00004' or 'Best'
  '''
  try:
    counts = SS.N
  except AttributeError:
    counts = SS.SumWordCounts
  assert counts.ndim == 1
  counts = np.asarray(counts, dtype=np.float32)
  np.maximum(counts, 0, out=counts)
  outmatfile = os.path.join(fpath, prefix + 'Counts.mat' )
  scipy.io.savemat(outmatfile, dict(N=counts), oned_as='row')