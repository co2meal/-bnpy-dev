'''
FromTruth.py

Initialize params of a bnpy model using "ground truth" side information,
such as human annotations 

These are provided within a Data object, as a "TrueLabels" field
'''
import numpy as np

def init_global_params(hmodel, Data, initname=None, seed=0, nRep=2, **kwargs):
  ''' Initialize (in-place) the global params of the given hmodel
      using the true labels associated with the Data

      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : string name for the routine to use
                 'truelabels' or 'repeattruelabels'
  '''
  TrueLabels = Data.TrueLabels
  uniqueLabels = np.unique(TrueLabels)
  Ktrue = len(uniqueLabels)
  PRNG = np.random.RandomState(seed)
  if initname == 'truelabels':
    resp = np.zeros((Data.nObs, Ktrue))
    for k in range(Ktrue):
      mask = TrueLabels == uniqueLabels[k]
      resp[mask,k] = 1.0
  elif initname == 'repeattruelabels':
    resp = np.zeros((Data.nObs, nRep*Ktrue)) 
    for k in range(Ktrue):
      mask = TrueLabels == uniqueLabels[k]
      if np.sum(mask) < nRep:
        raise ValueError('Not enough examples of label %d' % (uniqueLabels[k]))
      # Shuffle list in place!
      obsIDs = np.flatnonzero(mask)
      PRNG.shuffle(obsIDs)
      curLoc = 0
      L = len(obsIDs)/nRep
      for r in range(nRep):
        targetIDs = obsIDs[curLoc:curLoc+L]
        resp[targetIDs, k + r*Ktrue] = 1.0
        curLoc += L
  else:
    raise NotImplementedError('Unknown initname: %s' % (initname))

  LP = dict(resp=resp)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
